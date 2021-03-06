import pandas as pd
import numpy as np
from numpy.linalg import inv

class BlackLitterman:

    def __init__(self):

        self.weights = None
        self.implied_equilibrium_returns = None
        self.posterior_expected_returns = None
        self.posterior_covariance = None

    def allocate(self, covariance, market_capitalised_weights, investor_views, pick_list, omega=None, risk_aversion=2.5, tau=0.05,
                 omega_method='prior_variance', view_confidences=None, asset_names=None):

        # Initial check of inputs.
        self._error_checks(investor_views, pick_list, omega_method, view_confidences)

        num_assets = len(market_capitalised_weights)
        num_views = len(investor_views)
        if asset_names is None:
            if covariance is not None and isinstance(covariance, pd.DataFrame):
                asset_names = covariance.columns
            else:
                asset_names = list(map(str, range(num_assets)))
        covariance, market_capitalised_weights, investor_views = self._pre_process_inputs(covariance, market_capitalised_weights, investor_views)

        # Calculate the implied excess market equilibrium returns using reverse optimisation trick
        self.implied_equilibrium_returns = self._calculate_implied_equilibrium_returns(risk_aversion, covariance, market_capitalised_weights)

        # Create the pick matrix (P) from user specified assets involved in the views
        pick_matrix = self._create_pick_matrix(num_views, num_assets, pick_list, asset_names)
        print(pick_matrix)

        # Build the covariance matrix of errors in investor views (omega)
        if omega is None:
            omega = self._calculate_omega(covariance, tau, pick_matrix, view_confidences, omega_method)
        omega = np.array(np.reshape(omega, (num_views, num_views)))

        # BL expected returns
        self.posterior_expected_returns = self._calculate_posterior_expected_returns(covariance, tau, pick_matrix, omega, investor_views)

        # BL covariance
        self.posterior_covariance = self._calculate_posterior_covariance(covariance, tau, pick_matrix, omega)

        # Get optimal weights
        self.weights = self._calculate_max_sharpe_weights()

        # Post processing
        self._post_processing(asset_names)

    @staticmethod
    def _pre_process_inputs(covariance, market_capitalised_weights, investor_views):
        """
        Initial preprocessing of inputs.

        :param covariance: (pd.DataFrame/Numpy matrix) The covariance matrix of asset returns.
        :param market_capitalised_weights: (Numpy array/Python list) List of market capitalised weights of assets.
        :param investor_views: (Numpy array/Python list) User-specified list of views expressed in the form of percentage excess returns.
        :return: (Numpy matrix, Numpy array, Numpy matrix) Preprocessed inputs.
        """

        investor_views = np.array(np.reshape(investor_views, newshape=(len(investor_views), 1)))
        market_capitalised_weights = np.array(np.reshape(market_capitalised_weights, newshape=(len(market_capitalised_weights), 1)))
        if isinstance(covariance, pd.DataFrame):
            covariance = covariance.values

        return covariance, market_capitalised_weights, investor_views

    def _calculate_max_sharpe_weights(self):

        weights = inv(self.posterior_covariance).dot(self.posterior_expected_returns.T)
        weights /= sum(weights)
        return weights

    def _calculate_posterior_expected_returns(self, covariance, tau, pick_matrix, omega, investor_views):
        """
        Calculate Black-Litterman expected returns from investor views.

        :param covariance: (pd.DataFrame/Numpy matrix) The covariance matrix of asset returns.
        :param tau: (float) Constant of proportionality.
        :param pick_matrix: (Numpy matrix) Matrix specifying which assets involved in the respective view.
        :param omega: (Numpy matrix) Diagonal matrix of variance in investor views.
        :param investor_views: (Numpy array/Python list) User-specified list of views expressed in the form of percentage excess returns.
        :return: (Numpy array) Posterior expected returns.
        """

        posterior_expected_returns = self.implied_equilibrium_returns + (tau * covariance).dot(pick_matrix.T).\
            dot(inv(pick_matrix.dot(tau * covariance).dot(pick_matrix.T) + omega).dot(investor_views - pick_matrix.dot(self.implied_equilibrium_returns)))
        posterior_expected_returns = posterior_expected_returns.reshape(1, -1)
        return posterior_expected_returns

    @staticmethod
    def _calculate_posterior_covariance(covariance, tau, pick_matrix, omega):
        """
        Calculate Black-Litterman covariance of asset returns from investor views.

        :param covariance: (pd.DataFrame/Numpy matrix) The covariance matrix of asset returns.
        :param tau: (float) Constant of proportionality
        :param pick_matrix: (Numpy matrix) Matrix specifying specifying which assets involved in the respective view.
        :param omega: (Numpy matrix) Diagonal matrix of variance in investor views.
        :return: (Numpy array) Posterior covariance of asset returns.
        """

        posterior_covariance = covariance + (tau * covariance) - (tau * covariance).dot(pick_matrix.T).\
            dot(inv(pick_matrix.dot(tau * covariance).dot(pick_matrix.T) + omega)).dot(pick_matrix).dot(tau * covariance)
        return posterior_covariance

    @staticmethod
    def _calculate_implied_equilibrium_returns(risk_aversion, covariance, market_capitalised_weights):
        """
        Calculate the CAPM implied equilibrium market weights using the reverse optimisation trick.

        :param risk_aversion: (float) Quantifies the risk averse nature of the investor - a higher value means more risk averse and vice-versa.
        :param covariance: (pd.DataFrame/Numpy matrix) The covariance matrix of asset returns.
        :param market_capitalised_weights: (Numpy array/Python list) List of market capitalised weights of portfolio assets.
        :return: (Numpy array) Market equilibrium weights.
        """

        return risk_aversion * covariance.dot(market_capitalised_weights)

    @staticmethod
    def _create_pick_matrix(num_views, num_assets, pick_list, asset_names):
        """
        Calculate the picking matrix that specifies which assets are involved in the accompanying views.

        :param num_views: (int) Number of views.
        :param num_assets: (int) Number of assets in the portfolio.
        :param pick_list: (Numpy array/Python list) List of dictionaries specifying which assets involved in the respective view.
        :param asset_names: (Numpy array/Python list) A list of strings specifying the asset names.
        :return: (Numpy matrix) Picking matrix.
        """

        pick_matrix = np.zeros((num_views, num_assets))
        pick_matrix = pd.DataFrame(pick_matrix, columns=asset_names)
        for view_index, pick_dict in enumerate(pick_list):
            assets = list(pick_dict.keys())
            values = list(pick_dict.values())
            pick_matrix.loc[view_index, assets] = values
        return pick_matrix.values

    def _calculate_omega(self, covariance, tau, pick_matrix, view_confidences, omega_method):
        """
        Calculate the omega matrix - uncertainty in investor views.

        :param covariance: (pd.DataFrame/Numpy matrix) The covariance matrix of asset returns.
        :param tau: (float) Constant of proportionality
        :param pick_matrix: (Numpy matrix) Matrix specifying specifying which assets involved in the respective view.
        :param view_confidences: (Numpy array/Python list) Use supplied confidences for the views. The confidences are specified
                                                           in percentages e.g. 0.05, 0.4, 0.9 etc....
        :param omega_method: (str) The type of method to use for calculating the omega matrix.
        :return: (Numpy matrix) Omega matrix.
        """

        if omega_method == 'prior_variance':
            omega = pick_matrix.dot((tau * covariance).dot(pick_matrix.T))
        else:
            omega = self._calculate_idzorek_omega(covariance, view_confidences, pick_matrix)
        omega = np.diag(np.diag(omega))
        return omega

    @staticmethod
    def _calculate_idzorek_omega(covariance, view_confidences, pick_matrix):
        """
        Calculate the Idzorek omega matrix by taking into account user-supplied confidences in the views.

        :param covariance: (pd.DataFrame/Numpy matrix) The covariance matrix of asset returns.
        :param view_confidences: (Numpy array/Python list) Use supplied confidences for the views. The confidences are specified
                                                           in percentages e.g. 0.05, 0.4, 0.9 etc....
        :param pick_matrix: (Numpy matrix) Matrix specifying specifying which assets involved in the respective view.
        :return: (Numpy matrix) Idzorek Omega matrix.
        """

        view_confidences = np.array(np.reshape(view_confidences, (1, covariance.shape[0])))
        alpha = (1 - view_confidences) / view_confidences
        omega = alpha * pick_matrix.dot(covariance).dot(pick_matrix.T)
        return omega

    def _post_processing(self, asset_names):
        """
        Final post processing of weights, expected returns and covariance matrix.

        :param asset_names: (Numpy array/Python list) A list of strings specifying the asset names.
        """

        self.weights = self.weights.T
        self.weights = pd.DataFrame(self.weights, columns=asset_names)
        self.implied_equilibrium_returns = pd.DataFrame(self.implied_equilibrium_returns.T, columns=asset_names)
        self.posterior_expected_returns = pd.DataFrame(self.posterior_expected_returns, columns=asset_names)
        self.posterior_covariance = pd.DataFrame(self.posterior_covariance, columns=asset_names, index=asset_names)

    @staticmethod
    def _error_checks(investor_views, pick_list, omega_method, view_confidences):
        """
        Perform initial warning checks.

        :param investor_views: (Numpy array/Python list) User-specified list of views expressed in the form of percentage excess returns.
        :param pick_list: (Numpy array/Python list) List of dictionaries specifying which assets involved in the respective view.
        :param omega_method: (str) The type of method to use for calculating the omega matrix.
        :param view_confidences: (Numpy array/Python list) Use supplied confidences for the views. The confidences are specified
                                                           in percentages e.g. 0.05, 0.4, 0.9 etc....
        """

        if len(investor_views) != len(pick_list):
            raise ValueError("The number of views does not match the number of elements in the pick list.")

        if omega_method not in {'prior_variance', 'user_confidences'}:
            raise ValueError("Unknown omega method specified. Supported strings are - prior_variance, user_confidences")

        if omega_method == 'user_confidences':
            if view_confidences is None:
                raise ValueError("View confidences are required for calculating the Idzorek omega matrix.")

            if len(investor_views) != len(view_confidences):
                raise ValueError("The number of views does not match the number of view confidences specified.")

            for confidence in view_confidences:
                if confidence < 0:
                    raise ValueError("View confidence cannot be negative. Please specify a confidence value > 0.")
