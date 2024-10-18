import pandas as pd
from scipy import stats

class StatisticalTests:
    def __init__(self, df=None):
        """Initialize the class with a DataFrame."""
        if df is not None:
            self.data = df
        else:
            raise ValueError("DataFrame cannot be None.")

    # Method to perform Chi-Square test
    def perform_chisquare(self, categorical_var1, categorical_var2):
        """Perform Chi-Square test on two categorical variables."""
        contingency_table = pd.crosstab(self.data[categorical_var1], self.data[categorical_var2])
        stat, p, dof, expected = stats.chi2_contingency(contingency_table)
        print(f"Chi-Square Statistic: {stat}, p-value: {p}")
        return stat, p

    # Method to perform t-Test (assuming this exists in gex4)
    def perform_tests(self, continuous_var, categorical_var):
        """Perform t-Test or other statistical tests."""
        # Implementation for t-Test or other tests
        print(f"Performing t-Test between {continuous_var} and {categorical_var}")
        # Example test using independent t-test (depending on the data):
        groups = self.data[categorical_var].unique()
        if len(groups) == 2:
            group1 = self.data[self.data[categorical_var] == groups[0]][continuous_var]
            group2 = self.data[self.data[categorical_var] == groups[1]][continuous_var]
            stat, p = stats.ttest_ind(group1, group2)
            print(f"t-Statistic: {stat}, p-value: {p}")
        else:
            print(f"{categorical_var} must have exactly 2 groups for a t-Test.")

    # Method to perform regression
    def perform_regression(self, dependent_var, independent_var):
        """Perform regression analysis."""
        from statsmodels.formula.api import ols
        formula = f'{dependent_var} ~ {independent_var}'
        model = ols(formula, data=self.data).fit()
        print(model.summary())
