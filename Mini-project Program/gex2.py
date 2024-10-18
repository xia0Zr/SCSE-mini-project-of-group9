import pandas as pd
import matplotlib.pyplot as plt

class DataInspection:
    def __init__(self, df=None):
        """Initialize the class with a DataFrame."""
        self.df = df

    def handle_missing_values(self, column_name):
        """Handle missing values based on column type."""
        missing_count = self.df[column_name].isna().sum()
        if missing_count < len(self.df) * 0.5:
            if pd.api.types.is_numeric_dtype(self.df[column_name]):
                self.df[column_name].fillna(self.df[column_name].mean(), inplace=True)
            else:
                self.df[column_name].fillna(self.df[column_name].mode()[0], inplace=True)
        else:
            self.df.drop(column_name, axis=1, inplace=True)

    def check_data_types(self, column_name):
        """Check and convert data types if necessary."""
        if self.df[column_name].dtype == 'object':
            self.df[column_name] = pd.to_numeric(self.df[column_name], errors='coerce')

    def classify_and_calculate(self, column_name):
        """Calculate statistics based on the column type."""
        if pd.api.types.is_numeric_dtype(self.df[column_name]):
            median_val = self.df[column_name].median()
            self.plot_boxplot(column_name)  # Ensure boxplot is called
            return median_val
        else:
            mode_val = self.df[column_name].mode()[0]
            self.plot_bar_chart(column_name)  # Ensure bar chart is called
            return mode_val

    def plot_histogram(self, column_name):
        """Plot a histogram of a numeric column."""
        if pd.api.types.is_numeric_dtype(self.df[column_name]):
            plt.hist(self.df[column_name], bins=30, alpha=0.7, color='blue')
            plt.title(f'Histogram of {column_name}')
            plt.xlabel(column_name)
            plt.ylabel('Frequency')
            plt.show()

    def plot_boxplot(self, column_name):
        """Plot a boxplot of a numeric column."""
        if pd.api.types.is_numeric_dtype(self.df[column_name]):
            plt.boxplot(self.df[column_name])
            plt.title(f'Boxplot of {column_name}')
            plt.ylabel(column_name)
            plt.show()

    def plot_scatter(self, x_column, y_column):
        """Plot a scatter plot of two numeric columns."""
        if pd.api.types.is_numeric_dtype(self.df[x_column]) and pd.api.types.is_numeric_dtype(self.df[y_column]):
            plt.scatter(self.df[x_column], self.df[y_column], alpha=0.7)
            plt.title(f'Scatter Plot of {x_column} vs {y_column}')
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            plt.show()

    def plot_bar_chart(self, column_name):
        """Plot a bar chart of categorical data."""
        if not pd.api.types.is_numeric_dtype(self.df[column_name]):
            self.df[column_name].value_counts().plot(kind='bar')
            plt.title(f'Bar Chart of {column_name}')
            plt.xlabel(column_name)
            plt.ylabel('Frequency')
            plt.show()

    def ask_for_scatterplot(self):
        """Ask user for two continuous columns to plot a scatter plot."""
        continuous_columns = [col for col in self.df.columns if pd.api.types.is_numeric_dtype(self.df[col])]
        if len(continuous_columns) < 2:
            print("Not enough continuous columns for a scatter plot.")
            return
        print("Select columns for scatter plot:")
        for i, col in enumerate(continuous_columns):
            print(f"{i + 1}: {col}")
        col1_index = int(input("Enter the index of the first column: ")) - 1
        col2_index = int(input("Enter the index of the second column: ")) - 1
        self.plot_scatter(continuous_columns[col1_index], continuous_columns[col2_index])

    def ask_for_correlation(self, numeric_cols):
        """Ask user for two numeric columns to calculate correlation."""
        print("Select columns for correlation:")
        for i, col in enumerate(numeric_cols):
            print(f"{i + 1}: {col}")
        col1_index = int(input("Enter the index of the first column: ")) - 1
        col2_index = int(input("Enter the index of the second column: ")) - 1
        return self.df[numeric_cols[col1_index]].corr(self.df[numeric_cols[col2_index]])

    def ask_for_std(self, numeric_cols):
        """Ask user for a numeric column to calculate standard deviation."""
        print("Select a column for standard deviation:")
        for i, col in enumerate(numeric_cols):
            print(f"{i + 1}: {col}")
        col_index = int(input("Enter the index of the column: ")) - 1
        return self.df[numeric_cols[col_index]].std()

    def ask_for_skewness(self, numeric_cols):
        """Ask user for a numeric column to calculate skewness."""
        print("Select a column for skewness:")
        for i, col in enumerate(numeric_cols):
            print(f"{i + 1}: {col}")
        col_index = int(input("Enter the index of the column: ")) - 1
        return self.df[numeric_cols[col_index]].skew()

    def ask_for_kurtosis(self, numeric_cols):
        """Ask user for a numeric column to calculate kurtosis."""
        print("Select a column for kurtosis:")
        for i, col in enumerate(numeric_cols):
            print(f"{i + 1}: {col}")
        col_index = int(input("Enter the index of the column: ")) - 1
        return self.df[numeric_cols[col_index]].kurt()
