# gex3.py
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

class DataAnalysis:
    def __init__(self, data):
        self.data = data

    def perform_anova(self, continuous_var, categorical_var):
        try:
            # Group the data by the categorical variable and extract the continuous data
            groups = [group[continuous_var].dropna().values for name, group in self.data.groupby(categorical_var)]
            
            # Perform the ANOVA test
            f_val, p_val = stats.f_oneway(*groups)
            
            # Display results
            print(f"ANOVA results: F-value = {f_val}, P-value = {p_val}")
            if p_val < 0.05:
                print(f"Since P-value ({p_val}) < 0.05, we reject the null hypothesis.")
            else:
                print(f"Since P-value ({p_val}) >= 0.05, we fail to reject the null hypothesis.")
            
            # Optionally, visualize the result
            self.visualize_anova(continuous_var, categorical_var)

        except KeyError:
            print(f"Error: One or both of the variables '{continuous_var}' or '{categorical_var}' were not found in the dataset.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def visualize_anova(self, continuous_var, categorical_var):
        try:
            # Visualizing ANOVA results with a boxplot
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=self.data[categorical_var], y=self.data[continuous_var])
            plt.title(f"Boxplot of {continuous_var} by {categorical_var}")
            plt.show()
        except Exception as e:
            print(f"Error during visualization: {e}")

    def check_normality(self, variable):
        # Check if a variable follows a normal distribution using a Q-Q plot
        from scipy import stats
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 6))
        stats.probplot(self.data[variable].dropna(), dist="norm", plot=plt)
        plt.title(f"Q-Q Plot for {variable}")
        plt.show()

    def calculate_skewness(self, variable):
        # Calculate and print skewness for a variable
        skewness = self.data[variable].skew()
        print(f"The skewness of {variable} is {skewness}")
        return skewness

    def perform_kruskal_wallis(self, continuous_var, categorical_var):
        try:
            # Group the data by the categorical variable and extract the continuous data
            groups = [group[continuous_var].dropna().values for name, group in self.data.groupby(categorical_var)]
            
            # Perform the Kruskal-Wallis test
            h_val, p_val = stats.kruskal(*groups)
            
            # Display results
            print(f"Kruskal-Wallis results: H-value = {h_val}, P-value = {p_val}")
            if p_val < 0.05:
                print(f"Since P-value ({p_val}) < 0.05, we reject the null hypothesis.")
            else:
                print(f"Since P-value ({p_val}) >= 0.05, we fail to reject the null hypothesis.")
        except KeyError:
            print(f"Error: One or both of the variables '{continuous_var}' or '{categorical_var}' were not found in the dataset.")
        except Exception as e:
            print(f"An error occurred: {e}")
