import gex2  # Data Inspection module
import gex3  # ANOVA module
import gex4  # t-Test, Chi-Square, Regression module
import gex5  # Sentiment Analysis module
import pandas as pd  # For handling the dataset
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels.api as sm

def perform_analysis(data, dataset_path):
    while True:
        # Step 1: Show analysis options
        print("\nChoose the analysis you want to perform:")
        print("1. Plot variable distribution")
        print("2. Conduct ANOVA")
        print("3. Conduct t-Test")
        print("4. Conduct Chi-Square Test")
        print("5. Conduct Regression")
        print("6. Sentiment Analysis")
        print("7. Quit")

        choice = input("Enter your choice (1-7): ")

        if choice == '1':
            # Variable distribution plot: Loop until user chooses to go back or quit
            while True:
                # Show available variables for plot distribution
                numeric_columns = data.select_dtypes(include='number').columns.tolist()
                print("\nFollowing variables are available for plot distribution:")
                for idx, var in enumerate(numeric_columns, 1):
                    print(f"{idx}. {var}")
                print(f"{len(numeric_columns) + 1}. BACK")
                print(f"{len(numeric_columns) + 2}. QUIT")

                # Ask for user input to select variable or go back
                selected = input("Enter the number of the variable you want to plot (or choose BACK/QUIT): ")
                try:
                    selected = int(selected)
                    if selected == len(numeric_columns) + 1:
                        break  # Go back to analysis options
                    elif selected == len(numeric_columns) + 2:
                        print("Exiting the program.")
                        exit()
                    elif 1 <= selected <= len(numeric_columns):
                        variable = numeric_columns[selected - 1]
                        # Plot the selected variable's distribution
                        inspector = gex2.DataInspection(dataset_path)
                        inspector.plot_histogram(variable)
                    else:
                        print("Invalid choice. Please enter a valid option.")
                except ValueError:
                    print("Invalid input. Please enter a valid number.")
            continue  # Return to the main analysis options menu

        elif choice == '2':
            # ANOVA analysis: select continuous variable and categorical variable
            numeric_columns = data.select_dtypes(include='number').columns.tolist()
            categorical_columns = data.select_dtypes(include=['category', 'object']).columns.tolist()

            continuous_var = select_variable(numeric_columns, "continuous")
            categorical_var = select_variable(categorical_columns, "categorical")

            # Check for normality using Q-Q plot
            if not check_normality(data[continuous_var]):
                print(f"{continuous_var} is not normally distributed. Performing Kruskal-Wallis Test instead.")
                perform_kruskal_wallis_test(data, continuous_var, categorical_var)
            else:
                perform_anova(data, continuous_var, categorical_var)
            continue  # Return to analysis options

        elif choice == '3':
            # t-Test: select continuous variable and categorical variable
            numeric_columns = data.select_dtypes(include='number').columns.tolist()
            categorical_columns = data.select_dtypes(include=['category', 'object']).columns.tolist()

            continuous_var = select_variable(numeric_columns, "continuous")
            categorical_var = select_variable(categorical_columns, "categorical")
            
            stats_test = gex4.StatisticalTests(data)
            stats_test.perform_tests(continuous_var, categorical_var)  # Perform t-Test
            continue  # Return to analysis options

        elif choice == '4':
            # Chi-Square Test: select two categorical variables
            categorical_columns = data.select_dtypes(include=['category', 'object']).columns.tolist()

            categorical_var1 = select_variable(categorical_columns, "first categorical")
            categorical_var2 = select_variable(categorical_columns, "second categorical")
            
            stats_test = gex4.StatisticalTests(data)
            stats_test.perform_chisquare(categorical_var1, categorical_var2)  # Perform Chi-Square test
            continue  # Return to analysis options

        elif choice == '5':
            # Regression: select two continuous variables
            numeric_columns = data.select_dtypes(include='number').columns.tolist()

            dependent_var = select_variable(numeric_columns, "dependent")
            independent_var = select_variable(numeric_columns, "independent")
            
            regression_test = gex4.StatisticalTests(data)
            regression_test.perform_regression(dependent_var, independent_var)  # Perform regression analysis
            continue  # Return to analysis options

        elif choice == '6':
            # Sentiment Analysis
            sentiment = gex5.SentimentAnalysis()
            sentiment.load_data(data)  # Load the data for sentiment analysis
            sentiment_type = input("Choose sentiment analysis type (1: VADER, 2: TextBlob, 3: DistilBERT): ")
            if sentiment_type == '1':
                sentiment.vader_sentiment_analysis()  # Perform VADER sentiment analysis
            elif sentiment_type == '2':
                sentiment.textblob_sentiment_analysis()  # Perform TextBlob sentiment analysis
            elif sentiment_type == '3':
                sentiment.distilbert_sentiment_analysis()  # Perform DistilBERT sentiment analysis
            continue  # Return to analysis options

        elif choice == '7':
            print("Exiting the program.")
            exit()  # Exit the program

        else:
            print("Invalid choice. Please select a valid option.")

# Helper function for variable selection
def select_variable(variable_list, variable_type):
    print(f"\nSelect the {variable_type} variable:")
    for idx, var in enumerate(variable_list, 1):
        print(f"{idx}. {var}")
    while True:
        try:
            choice = int(input(f"Enter the number of the {variable_type} variable: "))
            if 1 <= choice <= len(variable_list):
                return variable_list[choice - 1]  # Return selected variable
            else:
                print(f"Please enter a number between 1 and {len(variable_list)}.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

# Function to perform normality check using Q-Q plot
def check_normality(variable_data):
    sm.qqplot(variable_data, line='s')
    plt.title('Q-Q Plot for Normality Check')
    plt.show()

    # Perform Shapiro-Wilk test for normality
    stat, p = stats.shapiro(variable_data)
    print(f'Shapiro-Wilk Test: statistic={stat}, p-value={p}')
    
    # If p-value is less than 0.05, data is not normally distributed
    return p > 0.05

# Function to perform ANOVA
def perform_anova(data, continuous_var, categorical_var):
    print(f"Performing ANOVA on {continuous_var} and {categorical_var}...")
    anova = gex3.DataAnalysis(data)
    anova.perform_anova(continuous_var, categorical_var)

# Function to perform Kruskal-Wallis Test
def perform_kruskal_wallis_test(data, continuous_var, categorical_var):
    print(f"Performing Kruskal-Wallis Test on {continuous_var} and {categorical_var}...")
    
    groups = data[categorical_var].unique()
    group_data = [data[data[categorical_var] == group][continuous_var] for group in groups]
    
    # Perform Kruskal-Wallis Test
    stat, p = stats.kruskal(*group_data)
    print(f'Kruskal-Wallis Test: statistic={stat}, p-value={p}')
    
    if p < 0.05:
        print(f"There is a statistically significant difference between the groups for {continuous_var} (p < 0.05).")
    else:
        print(f"No statistically significant difference found (p >= 0.05).")

def main():
    # Get dataset path from the user
    dataset_path = input("Enter the path to your dataset (CSV format): ")

    try:
        # Load the CSV file
        data = pd.read_csv(dataset_path, encoding='ISO-8859-1')
        print("\nHere are the first 5 rows of the dataset:\n")
        print(data.head())

        # Start analysis
        perform_analysis(data, dataset_path)

    except UnicodeDecodeError as e:
        print(f"Error reading the file: {e}")
    except FileNotFoundError as e:
        print(f"Error: File not found. Please check the path: {e}")

if __name__ == "__main__":
    main()
