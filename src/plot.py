""" Used to Generate Plots """

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from matplotlib.axes import Axes

# --------------------------------------------------------------------------------------------------------------------------------------- #

class Diagnostics():
        """
        Produces diagnostic plots for the model
        """
        def __init__(self,
                     X: jax.Array,
                     y_true: jax.Array,
                     y_pred: jax.Array):
            self.X = X
            self.y_true = y_true
            self.y_pred = y_pred
            self.residuals = y_true - y_pred
            self.standardized_residuals = (self.residuals - self.residuals.mean())/self.residuals.std() 
            
        def residual_vs_fitted_plot(self,
                                    ax:Axes=None, 
                                    marker:str='+', 
                                    scatter_colour:str='black',
                                    line_colour:str='red'):
            """ 
            Outputs graph of residuals against fitted values.
            """
            if ax is None:
                fig, ax = plt.subplots(figsize=(5,5), facecolor='none')
                
            ax.set_title('Residuals vs Fitted Values')
            sns.residplot(x=self.y_pred, 
                          y=self.residuals, 
                          lowess=True, 
                          scatter_kws={'marker': marker, 'color': scatter_colour},
                          line_kws={'color': line_colour},
                          ax=ax)
            plt.xlabel('Fitted Values')
            plt.ylabel('Residuals')


        def qqplot(self, ax:Axes=None):
            """
            Outputs the graph of the Q-Q plot of the residuals
            """
            if ax is None:
                fig, ax = plt.subplots(figsize=(5,5), facecolor='none')

            ax.set_title('Q-Q Plot of Residuals')
            sm.qqplot(self.residuals, line='q', ax=ax)
            plt.tight_layout()


        def standardized_residuals_histogram(self,
                                             ax:Axes=None,
                                             bins:int=45):
            """
            Outputs histogram of standardized residuals
            """
            if ax is None:
                fig, ax = plt.subplots(figsize=(5,5), facecolor='none')

            plt.figure(figsize=(5,5), facecolor='none')
            ax.set_title('Histogram of Standardized Residuals')
            ax.hist(self.standardized_residuals, bins=bins)


        def scale_location_plot(self, 
                                ax:Axes=None,
                                marker:str='+', 
                                scatter_colour:str='black',
                                line_colour:str='red'):
            """
            Outputs graph of Scale-Location Plot
            """
            if ax is None:
                fig, ax = plt.subplots(figsize=(5,5), facecolor='none')

            # SCALE-LOCATION PLOT
            ax.set_title('Scale-Location Plot')
            ax.scatter(self.y_pred, jnp.sqrt(self.standardized_residuals), marker=marker, color=scatter_colour)
            sns.regplot(x=self.y_pred, 
                        y=jnp.sqrt(self.standardized_residuals), 
                        lowess=True, 
                        scatter=False, 
                        color=line_colour,
                        ax=ax)
            plt.xlabel('Fitted Values')
            plt.ylabel('sqrt(Standardized Residuals)')


        def __hat_matrix(self, X:jax.Array):
            """
            Returns hat matrix for X
            """
            return X @ jnp.linalg.inv(X.T @ X) @ X.T
        
        def residuals_vs_leverage_plot(self,
                                       ax:Axes=None,
                                       levels:list=[0.5,1,2],
                                       colours:list=['orange', 'red', 'purple'],):
            """
            Outputs plot of residuals against leverage.
            """
            # ----- DEFINING ----- #
        
            P = self.__hat_matrix(self.X)
            r = self.X.shape[1] 

            leverages = jnp.array([P[i,i] for i in range(P.shape[0])])

            # ----- PLOTTING ----- #

            if ax is None:
                fig, ax = plt.subplots(figsize=(5,5), facecolor='none')

            ax.set_title('Residuals vs Leverage')
            ax.set_xlabel('Leverage')
            ax.set_ylabel('Standardized Residuals')

            ax.scatter(leverages, self.standardized_residuals, color='black', marker='+')

            standardized_residuals_grid, leverages_grid = jnp.meshgrid(
                jnp.linspace(self.standardized_residuals.min()-30, self.standardized_residuals.max()+30, 100),
                jnp.linspace(leverages.min()-1e-3, leverages.max()+1e-3, 100)
            )

            sns.regplot(x=leverages, 
                        y=self.standardized_residuals, 
                        scatter=False,
                        line_kws={'color': 'red'},
                        lowess=True,
                        ax=ax)

            cooks_dist_grid = (standardized_residuals_grid**2)/r * leverages_grid/(1-leverages_grid)

            ax.contour(leverages_grid, standardized_residuals_grid, cooks_dist_grid, levels=levels, colors=colours, alpha=0.25)
                

        def diagnostic_plots(self,
                             figsize:tuple=(12,12)):
            """ 
            Outputs a 2x2 grid of the diagnostic plots
            """
            fig, axs = plt.subplots(2, 2, figsize=figsize)

            plt.sca(axs[0,0])
            self.residual_vs_fitted_plot(axs[0,0])
            plt.sca(axs[0,1])
            self.qqplot(axs[0,1])
            plt.sca(axs[1,0])
            self.scale_location_plot(axs[1,0])
            plt.sca(axs[1,1])
            self.residuals_vs_leverage_plot(axs[1,1])

            fig.subplots_adjust(hspace=0.2, wspace=0.2)

            plt.tight_layout()
            plt.show()
            