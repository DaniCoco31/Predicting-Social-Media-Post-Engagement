# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pickle
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Function to remove outliers using IQR
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Normalize the data
def normalize_data(df, column):
    scaler = MinMaxScaler()
    df[column] = scaler.fit_transform(df[[column]])
    return df

def load_and_merge_datasets(base_path, embedded_paths, sample_size=None, output_path=None):
    """
    Loads a base dataset and multiple embedded datasets, merges them based on 'id' and 'description',
    optionally samples the merged dataset, and saves it to a CSV file.

    :param base_path: str, the file path for the base dataset.
    :param embedded_paths: list, a list of file paths for the embedded datasets.
    :param sample_size: int, optional, the number of samples to take from the merged dataset (default is None, no sampling).
    :param output_path: str, optional, the path to save the merged dataset (default is None, no saving).
    :return: DataFrame, the merged (and possibly sampled) dataset.
    """
    # Load the base dataset
    try:
        base_data = pd.read_csv(base_path)
        print("Base Data Loaded:")
        #print(base_data.head(2))
        print("Shape of the base data:", base_data.shape)
    except Exception as e:
        print(f"Error loading base data from {base_path}: {e}")
        return None

    # Load and concatenate the embedded datasets
    embedded_data = []
    for path in embedded_paths:
        try:
            df = pd.read_csv(path)
            embedded_data.append(df)
            print(f"Data loaded from {path}")
        except Exception as e:
            print(f"Error loading data from {path}: {e}")

    if not embedded_data:
        print("No embedded data to merge.")
        return base_data

    embed_df = pd.concat(embedded_data, ignore_index=True)
    print("Embedded Data Loaded and Concatenated.")
    
    # Merge the datasets on 'id' and 'description'
    if 'id' in base_data.columns and 'description' in base_data.columns:
        merged_data = base_data.merge(embed_df, on=['id', 'description'])
        print("Data merged successfully. Shape of the merged data:", merged_data.shape)
    else:
        print("Required columns for merging not found.")
        return base_data

    # Sample the data if a sample size is specified
    if sample_size is not None and sample_size < len(merged_data):
        merged_data = merged_data.sample(n=sample_size)
        print(f"Data sampled to {sample_size} entries.")

    # Save the merged (and possibly sampled) data to a CSV file if an output path is specified
    if output_path:
        merged_data.to_csv(output_path, index=False)
        print(f"Merged data saved to {output_path}")

    return merged_data


def load_and_merge_datasets_sql(base_path, embedded_paths, sample_size=None, output_path=None):
    """
    Loads a base dataset and multiple embedded datasets, merges them based on 'id' and 'description',
    optionally samples the merged dataset, and saves it to a CSV file.

    :param base_path: str, the file path for the base dataset.
    :param embedded_paths: list, a list of file paths for the embedded datasets.
    :param sample_size: int, optional, the number of samples to take from the merged dataset (default is None, no sampling).
    :param output_path: str, optional, the path to save the merged dataset (default is None, no saving).
    :return: DataFrame, the merged (and possibly sampled) dataset.
    """
    # Load the base dataset
    try:
        base_data = pd.read_csv(base_path)
        print("Base Data Loaded:")
        #print(base_data.head(2))
        print("Shape of the base data:", base_data.shape)
    except Exception as e:
        print(f"Error loading base data from {base_path}: {e}")
        return None

    # Load and concatenate the embedded datasets
    embedded_data = []
    for path in embedded_paths:
        try:
            df = pd.read_csv(path)
            embedded_data.append(df)
            print(f"Data loaded from {path}")
        except Exception as e:
            print(f"Error loading data from {path}: {e}")

    if not embedded_data:
        print("No embedded data to merge.")
        return base_data

    embed_df = pd.concat(embedded_data, ignore_index=True)
    print("Embedded Data Loaded and Concatenated.")
    
    # Merge the datasets on 'id' and 'description'
    if 'id' in base_data.columns and 'description' in base_data.columns:
        merged_data = base_data.merge(embed_df, on=['id', 'description'])
        print("Data merged successfully. Shape of the merged data:", merged_data.shape)
    else:
        print("Required columns for merging not found.")
        return base_data

    # Sample the data if a sample size is specified
    if sample_size is not None and sample_size < len(merged_data):
        merged_data = merged_data.sample(n=sample_size)
        print(f"Data sampled to {sample_size} entries.")

    # Save the merged (and possibly sampled) data to a CSV file if an output path is specified
    if output_path:
        merged_data.to_csv(output_path, index=False)
        print(f"Merged data saved to {output_path}")

    return merged_data

# Example of function usage:
#base_path = '../Data/DF_complete/base.csv'
#embedded_paths = ['../Data/DF_embedded/df_6-embed.csv']
#merged_data = load_and_merge_datasets(base_path, embedded_paths, sample_size=10000, output_path='../Data/Clean-Data/merged_data.csv')


def clean_and_wrangle_data(data_collected, data_cleaned):
    """
    Loads merged data, handles missing values, filters data based on certain criteria,
    adds new columns, removes outliers, and saves the cleaned data to a CSV file.

    :param data_collected: str, the file path to the merged data CSV.
    :param data_cleaned: str, the file path where the cleaned data will be saved.
    
    :return: None, outputs a CSV file with the cleaned data.
    """

    # Load the merged data from the CSV file
    df = pd.read_csv(data_collected)
    print("Merged Data Loaded:")
    #print(df.head(2))  # Show the first two rows to confirm data is loaded
    print("Shape of the merged data:", df.shape)

    # Handling Missing Values
    missing_values = df.isnull().sum()
    print("Missing values in each column:\n", missing_values)
    df.dropna(subset=['description'], inplace=True)  # Drop rows where 'description' is missing
    print("Data after handling missing values:")
    print(df.shape)

    # Data Filtering (uncomment to activate filtering logic)
    # df = df[(df['followers'] > 500) & (df['followers'] < 1500)]
    # df = df[df['num_posts'] >= 100]
    # df = df[(df['description'].apply(len) >= 50) & (df['description'].apply(len) <= 200)]
    print("Data after filtering:")
    print(df.shape)

    # Adding a new column for the length of the description
    df['description_length'] = df['description'].apply(len)

    # Removing outliers based on the IQR strategy for multiple columns
    df = remove_outliers(df, 'followers')
    df = remove_outliers(df, 'interactions')
    df = remove_outliers(df, 'num_posts')
    df = remove_outliers(df, 'description_length')
    print("Data after removing outliers:")
    print(df.shape)

    # Apply Log Transformation
    df['followers_trans'] = np.log1p(df['followers'])
    df['interactions_trans'] = np.log1p(df['interactions'])
    df['num_posts_trans'] = np.log1p(df['num_posts'])
    df['description_length_trans'] = np.log1p(df['description_length'])

    # Saving the cleaned data to the specified output CSV file
    df.to_csv(data_cleaned, index=False)
    print(f"Cleaned data saved to {data_cleaned}")
    

# Example usage
#input_path = '../Data/Clean-Data/merged_data.csv'
#data_cleaned = '../Data/Clean-Data/cleaned_data.csv'
#clean_and_wrangle_data(data_collected, data_cleaned)


def perform_eda(file_path, output_path):
    """
    Performs exploratory data analysis on the provided dataset to understand relationships 
    between different variables, with a focus on interactions. This includes generating 
    distribution plots, a correlation heatmap, and various charts to explore data characteristics.
    
    :param file_path: str, path to the cleaned data file.
    :param output_path: str, path to save the data after performing EDA.
    :return: None
    """

    # Load the cleaned data
    df = pd.read_csv(file_path)
    print("Cleaned Data Loaded:")
    #print(df.head(2))
    print("Shape of the cleaned data:", df.shape)

    # Basic Statistical Analysis
    print("Basic Statistical Descriptions:")
    print(df[['followers', 'interactions', 'num_posts', 'description_length']].describe())

    # Set plot style
    sns.set(style="whitegrid")
    sns.set_palette('viridis')

    # Distribution of interactions
    sns.histplot(df['interactions'], bins=30, kde=True)
    plt.title('Distribution of Interactions')
    plt.show()
    print("This plot shows the distribution of interaction counts across posts, helping to identify common engagement levels and outliers.")

    # Distribution of followers
    sns.histplot(df['followers'], bins=30, kde=True)
    plt.title('Distribution of Followers')
    plt.show()
    print("The distribution of followers helps us understand the range and commonality of follower counts within the dataset.")

    # Distribution of day of the week
    sns.countplot(x='day_of_week', data=df, palette='viridis')
    plt.title('Distribution of Day of the Week')
    plt.show()
    print("This count plot provides insight into the activity or posting frequency by day of the week, which can influence engagement.")

    # Distribution of time of the day
    time_of_day = df['time_of_day'].value_counts()
    time_of_day.plot.pie(autopct='%1.1f%%', colors=sns.color_palette('viridis', len(time_of_day)))
    plt.title('Time of Day Distribution')
    plt.ylabel('')
    plt.show()
    print("The pie chart of time of day shows when posts are typically made, which is crucial for timing strategies in social media marketing.")

    # Distribution of business account status
    is_business_account = df['is_business_account'].value_counts()
    is_business_account.plot.pie(autopct='%1.1f%%', colors=sns.color_palette('viridis', len(is_business_account)))
    plt.title('Business Account Distribution')
    plt.ylabel('')
    plt.show()
    print("Understanding the proportion of business accounts can help tailor content and marketing strategies to the right audience.")

    # Distribution of the length of descriptions
    sns.histplot(df['description'].apply(len), bins=30, kde=True)
    plt.title('Distribution of Length of Description')
    plt.show()
    print("This histogram shows how the length of descriptions varies, which can affect both SEO and user engagement.")

    # Correlation heatmap
    df_heatmap = df[['followers', 'following', 'interactions', 'is_business_account', 'num_posts', 'description_length']]
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_heatmap.corr(), annot=True, cmap='viridis', fmt='.2f')
    plt.title('Heatmap of Data Correlations')
    plt.show()
    print("The heatmap illustrates the relationships between numeric variables, highlighting how each variable might affect interactions.")

    # Analyzing interactions with scatter plots
    sns.scatterplot(x='description_length', y='interactions', data=df)
    plt.title('Description Length vs. Interactions')
    plt.show()
    print("This scatter plot explores the relationship between the length of descriptions and user interactions.")

    sns.scatterplot(x='followers', y='interactions', data=df)
    plt.title('Followers vs. Interactions')
    plt.show()
    print("This plot indicates how follower count potentially influences the level of interactions on posts.")

    sns.scatterplot(x='num_posts', y='interactions', data=df)
    plt.title('Posts vs. Interactions')
    plt.show()
    print("Analyzing the correlation between the number of posts and interactions to determine if more frequent posting leads to more engagement.")

    # Save the data after EDA
    df.to_csv(output_path, index=False)
    print(f"EDA data saved to {output_path}")

# Example usage
#input_path = '../Data/Clean-Data/cleaned_data.csv'
#output_path = '../Data/Clean-Data/eda_data.csv'
#perform_eda(input_path, output_path)


def feature_engenieering(filepath, drop_columns, numerical_features, dummy_columns, output_path):
    """
    Prepares a dataset for machine learning modeling by performing one-hot encoding,
    scaling numerical features, and rearranging columns for the final modeling stage.

    Args:
    - filepath (str): Path to the CSV file containing the initial cleaned data.
    - drop_columns (list of str): Columns to be dropped from the dataset as they are not required for modeling.
    - numerical_features (list of str): List of csolumns that contain numerical features for scaling.
    - dummy_columns (list of str): List of categorical columns to be transformed using one-hot encoding.

    Returns:
    - df_to_model (DataFrame): The DataFrame ready for machine learning modeling.
    """

    # Step 1: Load the EDA data
    df = pd.read_csv(filepath)
    #print("EDA Data Loaded:")
    #print(df.head(2))
    print("Shape of the EDA data:", df.shape)

    # Step 2: One-Hot Encoding for Categorical Variables
    df = pd.get_dummies(df, columns=dummy_columns)
    #print("Data after one-hot encoding:")
    #print(df.head(2))
    #print("Columns in the data:", df.columns)

    #Step 2.5: Get the columns in order
    df = df[['id', 'description',
    'interactions', 'following', 'followers', 'num_posts','is_business_account',

    'category_arts_&_culture', 'category_business_&_entrepreneurs',
    'category_celebrity_&_pop_culture', 'category_diaries_&_daily_life',
    'category_family', 'category_fashion_&_style', 'category_film_tv_&_video',
    'category_fitness_&_health', 'category_food_&_dining', 'category_gaming',
    'category_learning_&_educational', 'category_music',
    'category_news_&_social_concern', 'category_other_hobbies',
    'category_relationships', 'category_science_&_technology', 'category_sports',
    'category_travel_&_adventure', 'category_youth_&_student_life',

    'day_of_week_Monday', 'day_of_week_Tuesday', 'day_of_week_Wednesday',
    'day_of_week_Thursday', 'day_of_week_Friday', 'day_of_week_Saturday',
    'day_of_week_Sunday',
    
    'time_of_day_early_morning',
    'time_of_day_morning', 
    'time_of_day_afternoon',
    'time_of_day_night', 

    'description_length',
    'followers_trans',
    'interactions_trans',
    'num_posts_trans',
    'description_length_trans',

    'embedded_0', 'embedded_1', 'embedded_2', 'embedded_3', 'embedded_4', 'embedded_5', 'embedded_6', 'embedded_7', 'embedded_8', 'embedded_9',
    'embedded_10', 'embedded_11', 'embedded_12', 'embedded_13', 'embedded_14', 'embedded_15', 'embedded_16', 'embedded_17', 'embedded_18', 'embedded_19',
    'embedded_20', 'embedded_21', 'embedded_22', 'embedded_23', 'embedded_24', 'embedded_25', 'embedded_26', 'embedded_27', 'embedded_28', 'embedded_29',
    'embedded_30', 'embedded_31', 'embedded_32', 'embedded_33', 'embedded_34', 'embedded_35', 'embedded_36', 'embedded_37', 'embedded_38', 'embedded_39',
    'embedded_40', 'embedded_41', 'embedded_42', 'embedded_43', 'embedded_44', 'embedded_45', 'embedded_46', 'embedded_47', 'embedded_48', 'embedded_49',
    'embedded_50', 'embedded_51', 'embedded_52', 'embedded_53', 'embedded_54', 'embedded_55', 'embedded_56', 'embedded_57', 'embedded_58', 'embedded_59',
    'embedded_60', 'embedded_61', 'embedded_62', 'embedded_63', 'embedded_64', 'embedded_65', 'embedded_66', 'embedded_67', 'embedded_68', 'embedded_69',
    'embedded_70', 'embedded_71', 'embedded_72', 'embedded_73', 'embedded_74', 'embedded_75', 'embedded_76', 'embedded_77', 'embedded_78', 'embedded_79',
    'embedded_80', 'embedded_81', 'embedded_82', 'embedded_83', 'embedded_84', 'embedded_85', 'embedded_86', 'embedded_87', 'embedded_88', 'embedded_89',
    'embedded_90', 'embedded_91', 'embedded_92', 'embedded_93', 'embedded_94', 'embedded_95', 'embedded_96', 'embedded_97', 'embedded_98', 'embedded_99',
    'embedded_100', 'embedded_101', 'embedded_102', 'embedded_103', 'embedded_104', 'embedded_105', 'embedded_106', 'embedded_107', 'embedded_108', 'embedded_109',
    'embedded_110', 'embedded_111', 'embedded_112', 'embedded_113', 'embedded_114', 'embedded_115', 'embedded_116', 'embedded_117', 'embedded_118', 'embedded_119',
    'embedded_120', 'embedded_121', 'embedded_122', 'embedded_123', 'embedded_124', 'embedded_125', 'embedded_126', 'embedded_127', 'embedded_128', 'embedded_129',
    'embedded_130', 'embedded_131', 'embedded_132', 'embedded_133', 'embedded_134', 'embedded_135', 'embedded_136', 'embedded_137', 'embedded_138', 'embedded_139',
    'embedded_140', 'embedded_141', 'embedded_142', 'embedded_143', 'embedded_144', 'embedded_145', 'embedded_146', 'embedded_147', 'embedded_148', 'embedded_149',
    'embedded_150', 'embedded_151', 'embedded_152', 'embedded_153', 'embedded_154', 'embedded_155', 'embedded_156', 'embedded_157', 'embedded_158', 'embedded_159',
    'embedded_160', 'embedded_161', 'embedded_162', 'embedded_163', 'embedded_164', 'embedded_165', 'embedded_166', 'embedded_167', 'embedded_168', 'embedded_169',
    'embedded_170', 'embedded_171', 'embedded_172', 'embedded_173', 'embedded_174', 'embedded_175', 'embedded_176', 'embedded_177', 'embedded_178', 'embedded_179',
    'embedded_180', 'embedded_181', 'embedded_182', 'embedded_183', 'embedded_184', 'embedded_185', 'embedded_186', 'embedded_187', 'embedded_188', 'embedded_189',
    'embedded_190', 'embedded_191', 'embedded_192', 'embedded_193', 'embedded_194', 'embedded_195', 'embedded_196', 'embedded_197', 'embedded_198', 'embedded_199',
    'embedded_200', 'embedded_201', 'embedded_202', 'embedded_203', 'embedded_204', 'embedded_205', 'embedded_206', 'embedded_207', 'embedded_208', 'embedded_209',
    'embedded_210', 'embedded_211', 'embedded_212', 'embedded_213', 'embedded_214', 'embedded_215', 'embedded_216', 'embedded_217', 'embedded_218', 'embedded_219',
    'embedded_220', 'embedded_221', 'embedded_222', 'embedded_223', 'embedded_224', 'embedded_225', 'embedded_226', 'embedded_227', 'embedded_228', 'embedded_229',
    'embedded_230', 'embedded_231', 'embedded_232', 'embedded_233', 'embedded_234', 'embedded_235', 'embedded_236', 'embedded_237', 'embedded_238', 'embedded_239',
    'embedded_240', 'embedded_241', 'embedded_242', 'embedded_243', 'embedded_244', 'embedded_245', 'embedded_246', 'embedded_247', 'embedded_248', 'embedded_249',
    'embedded_250', 'embedded_251', 'embedded_252', 'embedded_253', 'embedded_254', 'embedded_255', 'embedded_256', 'embedded_257', 'embedded_258', 'embedded_259',
    'embedded_260', 'embedded_261', 'embedded_262', 'embedded_263', 'embedded_264', 'embedded_265', 'embedded_266', 'embedded_267', 'embedded_268', 'embedded_269',
    'embedded_270', 'embedded_271', 'embedded_272', 'embedded_273', 'embedded_274', 'embedded_275', 'embedded_276', 'embedded_277', 'embedded_278', 'embedded_279',
    'embedded_280', 'embedded_281', 'embedded_282', 'embedded_283', 'embedded_284', 'embedded_285', 'embedded_286', 'embedded_287', 'embedded_288', 'embedded_289',
    'embedded_290', 'embedded_291', 'embedded_292', 'embedded_293', 'embedded_294', 'embedded_295', 'embedded_296', 'embedded_297', 'embedded_298', 'embedded_299',
    'embedded_300', 'embedded_301', 'embedded_302', 'embedded_303', 'embedded_304', 'embedded_305', 'embedded_306', 'embedded_307', 'embedded_308', 'embedded_309',
    'embedded_310', 'embedded_311', 'embedded_312', 'embedded_313', 'embedded_314', 'embedded_315', 'embedded_316', 'embedded_317', 'embedded_318', 'embedded_319',
    'embedded_320', 'embedded_321', 'embedded_322', 'embedded_323', 'embedded_324', 'embedded_325', 'embedded_326', 'embedded_327', 'embedded_328', 'embedded_329',
    'embedded_330', 'embedded_331', 'embedded_332', 'embedded_333', 'embedded_334', 'embedded_335', 'embedded_336', 'embedded_337', 'embedded_338', 'embedded_339',
    'embedded_340', 'embedded_341', 'embedded_342', 'embedded_343', 'embedded_344', 'embedded_345', 'embedded_346', 'embedded_347', 'embedded_348', 'embedded_349',
    'embedded_350', 'embedded_351', 'embedded_352', 'embedded_353', 'embedded_354', 'embedded_355', 'embedded_356', 'embedded_357', 'embedded_358', 'embedded_359',
    'embedded_360', 'embedded_361', 'embedded_362', 'embedded_363', 'embedded_364', 'embedded_365', 'embedded_366', 'embedded_367', 'embedded_368', 'embedded_369',
    'embedded_370', 'embedded_371', 'embedded_372', 'embedded_373', 'embedded_374', 'embedded_375', 'embedded_376', 'embedded_377', 'embedded_378', 'embedded_379',
    'embedded_380', 'embedded_381', 'embedded_382', 'embedded_383', 'embedded_384', 'embedded_385', 'embedded_386', 'embedded_387', 'embedded_388', 'embedded_389',
    'embedded_390', 'embedded_391', 'embedded_392', 'embedded_393', 'embedded_394', 'embedded_395', 'embedded_396', 'embedded_397', 'embedded_398', 'embedded_399',
    'embedded_400', 'embedded_401', 'embedded_402', 'embedded_403', 'embedded_404', 'embedded_405', 'embedded_406', 'embedded_407', 'embedded_408', 'embedded_409',
    'embedded_410', 'embedded_411', 'embedded_412', 'embedded_413', 'embedded_414', 'embedded_415', 'embedded_416', 'embedded_417', 'embedded_418', 'embedded_419',
    'embedded_420', 'embedded_421', 'embedded_422', 'embedded_423', 'embedded_424', 'embedded_425', 'embedded_426', 'embedded_427', 'embedded_428', 'embedded_429',
    'embedded_430', 'embedded_431', 'embedded_432', 'embedded_433', 'embedded_434', 'embedded_435', 'embedded_436', 'embedded_437', 'embedded_438', 'embedded_439',
    'embedded_440', 'embedded_441', 'embedded_442', 'embedded_443', 'embedded_444', 'embedded_445', 'embedded_446', 'embedded_447', 'embedded_448', 'embedded_449',
    'embedded_450', 'embedded_451', 'embedded_452', 'embedded_453', 'embedded_454', 'embedded_455', 'embedded_456', 'embedded_457', 'embedded_458', 'embedded_459',
    'embedded_460', 'embedded_461', 'embedded_462', 'embedded_463', 'embedded_464', 'embedded_465', 'embedded_466', 'embedded_467', 'embedded_468', 'embedded_469',
    'embedded_470', 'embedded_471', 'embedded_472', 'embedded_473', 'embedded_474', 'embedded_475', 'embedded_476', 'embedded_477', 'embedded_478', 'embedded_479',
    'embedded_480', 'embedded_481', 'embedded_482', 'embedded_483', 'embedded_484', 'embedded_485', 'embedded_486', 'embedded_487', 'embedded_488', 'embedded_489',
    'embedded_490', 'embedded_491', 'embedded_492', 'embedded_493', 'embedded_494', 'embedded_495', 'embedded_496', 'embedded_497', 'embedded_498', 'embedded_499',
    'embedded_500', 'embedded_501', 'embedded_502', 'embedded_503', 'embedded_504', 'embedded_505', 'embedded_506', 'embedded_507', 'embedded_508', 'embedded_509',
    'embedded_510', 'embedded_511', 'embedded_512', 'embedded_513', 'embedded_514', 'embedded_515', 'embedded_516', 'embedded_517', 'embedded_518', 'embedded_519',
    'embedded_520', 'embedded_521', 'embedded_522', 'embedded_523', 'embedded_524', 'embedded_525', 'embedded_526', 'embedded_527', 'embedded_528', 'embedded_529',
    'embedded_530', 'embedded_531', 'embedded_532', 'embedded_533', 'embedded_534', 'embedded_535', 'embedded_536', 'embedded_537', 'embedded_538', 'embedded_539',
    'embedded_540', 'embedded_541', 'embedded_542', 'embedded_543', 'embedded_544', 'embedded_545', 'embedded_546', 'embedded_547', 'embedded_548', 'embedded_549',
    'embedded_550', 'embedded_551', 'embedded_552', 'embedded_553', 'embedded_554', 'embedded_555', 'embedded_556', 'embedded_557', 'embedded_558', 'embedded_559',
    'embedded_560', 'embedded_561', 'embedded_562', 'embedded_563', 'embedded_564', 'embedded_565', 'embedded_566', 'embedded_567', 'embedded_568', 'embedded_569',
    'embedded_570', 'embedded_571', 'embedded_572', 'embedded_573', 'embedded_574', 'embedded_575', 'embedded_576', 'embedded_577', 'embedded_578', 'embedded_579',
    'embedded_580', 'embedded_581', 'embedded_582', 'embedded_583', 'embedded_584', 'embedded_585', 'embedded_586', 'embedded_587', 'embedded_588', 'embedded_589',
    'embedded_590', 'embedded_591', 'embedded_592', 'embedded_593', 'embedded_594', 'embedded_595', 'embedded_596', 'embedded_597', 'embedded_598', 'embedded_599',
    'embedded_600', 'embedded_601', 'embedded_602', 'embedded_603', 'embedded_604', 'embedded_605', 'embedded_606', 'embedded_607', 'embedded_608', 'embedded_609',
    'embedded_610', 'embedded_611', 'embedded_612', 'embedded_613', 'embedded_614', 'embedded_615', 'embedded_616', 'embedded_617', 'embedded_618', 'embedded_619',
    'embedded_620', 'embedded_621', 'embedded_622', 'embedded_623', 'embedded_624', 'embedded_625', 'embedded_626', 'embedded_627', 'embedded_628', 'embedded_629',
    'embedded_630', 'embedded_631', 'embedded_632', 'embedded_633', 'embedded_634', 'embedded_635', 'embedded_636', 'embedded_637', 'embedded_638', 'embedded_639',
    'embedded_640', 'embedded_641', 'embedded_642', 'embedded_643', 'embedded_644', 'embedded_645', 'embedded_646', 'embedded_647', 'embedded_648', 'embedded_649',
    'embedded_650', 'embedded_651', 'embedded_652', 'embedded_653', 'embedded_654', 'embedded_655', 'embedded_656', 'embedded_657', 'embedded_658', 'embedded_659',
    'embedded_660', 'embedded_661', 'embedded_662', 'embedded_663', 'embedded_664', 'embedded_665', 'embedded_666', 'embedded_667', 'embedded_668', 'embedded_669',
    'embedded_670', 'embedded_671', 'embedded_672', 'embedded_673', 'embedded_674', 'embedded_675', 'embedded_676', 'embedded_677', 'embedded_678', 'embedded_679',
    'embedded_680', 'embedded_681', 'embedded_682', 'embedded_683', 'embedded_684', 'embedded_685', 'embedded_686', 'embedded_687', 'embedded_688', 'embedded_689',
    'embedded_690', 'embedded_691', 'embedded_692', 'embedded_693', 'embedded_694', 'embedded_695', 'embedded_696', 'embedded_697', 'embedded_698', 'embedded_699',
    'embedded_700', 'embedded_701', 'embedded_702', 'embedded_703', 'embedded_704', 'embedded_705', 'embedded_706', 'embedded_707', 'embedded_708', 'embedded_709',
    'embedded_710', 'embedded_711', 'embedded_712', 'embedded_713', 'embedded_714', 'embedded_715', 'embedded_716', 'embedded_717', 'embedded_718', 'embedded_719',
    'embedded_720', 'embedded_721', 'embedded_722', 'embedded_723', 'embedded_724', 'embedded_725', 'embedded_726', 'embedded_727', 'embedded_728', 'embedded_729',
    'embedded_730', 'embedded_731', 'embedded_732', 'embedded_733', 'embedded_734', 'embedded_735', 'embedded_736', 'embedded_737', 'embedded_738', 'embedded_739',
    'embedded_740', 'embedded_741', 'embedded_742', 'embedded_743', 'embedded_744', 'embedded_745', 'embedded_746', 'embedded_747', 'embedded_748', 'embedded_749',
    'embedded_750', 'embedded_751', 'embedded_752', 'embedded_753', 'embedded_754', 'embedded_755', 'embedded_756', 'embedded_757', 'embedded_758', 'embedded_759',
    'embedded_760', 'embedded_761', 'embedded_762', 'embedded_763', 'embedded_764', 'embedded_765', 'embedded_766', 'embedded_767', 'embedded_768', 'embedded_769',
    'embedded_770', 'embedded_771', 'embedded_772', 'embedded_773', 'embedded_774', 'embedded_775', 'embedded_776', 'embedded_777', 'embedded_778', 'embedded_779',
    'embedded_780', 'embedded_781', 'embedded_782', 'embedded_783', 'embedded_784', 'embedded_785', 'embedded_786', 'embedded_787', 'embedded_788', 'embedded_789',
    'embedded_790', 'embedded_791', 'embedded_792', 'embedded_793', 'embedded_794', 'embedded_795', 'embedded_796', 'embedded_797', 'embedded_798', 'embedded_799',
    'embedded_800', 'embedded_801', 'embedded_802', 'embedded_803', 'embedded_804', 'embedded_805', 'embedded_806', 'embedded_807', 'embedded_808', 'embedded_809',
    'embedded_810', 'embedded_811', 'embedded_812', 'embedded_813', 'embedded_814', 'embedded_815', 'embedded_816', 'embedded_817', 'embedded_818', 'embedded_819',
    'embedded_820', 'embedded_821', 'embedded_822', 'embedded_823', 'embedded_824', 'embedded_825', 'embedded_826', 'embedded_827', 'embedded_828', 'embedded_829',
    'embedded_830', 'embedded_831', 'embedded_832', 'embedded_833', 'embedded_834', 'embedded_835', 'embedded_836', 'embedded_837', 'embedded_838', 'embedded_839',
    'embedded_840', 'embedded_841', 'embedded_842', 'embedded_843', 'embedded_844', 'embedded_845', 'embedded_846', 'embedded_847', 'embedded_848', 'embedded_849',
    'embedded_850', 'embedded_851', 'embedded_852', 'embedded_853', 'embedded_854', 'embedded_855', 'embedded_856', 'embedded_857', 'embedded_858', 'embedded_859',
    'embedded_860', 'embedded_861', 'embedded_862', 'embedded_863', 'embedded_864', 'embedded_865', 'embedded_866', 'embedded_867', 'embedded_868', 'embedded_869',
    'embedded_870', 'embedded_871', 'embedded_872', 'embedded_873', 'embedded_874', 'embedded_875', 'embedded_876', 'embedded_877', 'embedded_878', 'embedded_879',
    'embedded_880', 'embedded_881', 'embedded_882', 'embedded_883', 'embedded_884', 'embedded_885', 'embedded_886', 'embedded_887', 'embedded_888', 'embedded_889',
    'embedded_890', 'embedded_891', 'embedded_892', 'embedded_893', 'embedded_894', 'embedded_895', 'embedded_896', 'embedded_897', 'embedded_898', 'embedded_899',
    'embedded_900', 'embedded_901', 'embedded_902', 'embedded_903', 'embedded_904', 'embedded_905', 'embedded_906', 'embedded_907', 'embedded_908', 'embedded_909',
    'embedded_910', 'embedded_911', 'embedded_912', 'embedded_913', 'embedded_914', 'embedded_915', 'embedded_916', 'embedded_917', 'embedded_918', 'embedded_919',
    'embedded_920', 'embedded_921', 'embedded_922', 'embedded_923', 'embedded_924', 'embedded_925', 'embedded_926', 'embedded_927', 'embedded_928', 'embedded_929',
    'embedded_930', 'embedded_931', 'embedded_932', 'embedded_933', 'embedded_934', 'embedded_935', 'embedded_936', 'embedded_937', 'embedded_938', 'embedded_939',
    'embedded_940', 'embedded_941', 'embedded_942', 'embedded_943', 'embedded_944', 'embedded_945', 'embedded_946', 'embedded_947', 'embedded_948', 'embedded_949',
    'embedded_950', 'embedded_951', 'embedded_952', 'embedded_953', 'embedded_954', 'embedded_955', 'embedded_956', 'embedded_957', 'embedded_958', 'embedded_959',
    'embedded_960', 'embedded_961', 'embedded_962', 'embedded_963', 'embedded_964', 'embedded_965', 'embedded_966', 'embedded_967', 'embedded_968', 'embedded_969',
    'embedded_970', 'embedded_971', 'embedded_972', 'embedded_973', 'embedded_974', 'embedded_975', 'embedded_976', 'embedded_977', 'embedded_978', 'embedded_979',
    'embedded_980', 'embedded_981', 'embedded_982', 'embedded_983', 'embedded_984', 'embedded_985', 'embedded_986', 'embedded_987', 'embedded_988', 'embedded_989',
    'embedded_990', 'embedded_991', 'embedded_992', 'embedded_993', 'embedded_994', 'embedded_995', 'embedded_996', 'embedded_997', 'embedded_998', 'embedded_999',
    'embedded_1000', 'embedded_1001', 'embedded_1002', 'embedded_1003', 'embedded_1004', 'embedded_1005', 'embedded_1006', 'embedded_1007', 'embedded_1008', 'embedded_1009',
    'embedded_1010', 'embedded_1011', 'embedded_1012', 'embedded_1013', 'embedded_1014', 'embedded_1015', 'embedded_1016', 'embedded_1017', 'embedded_1018', 'embedded_1019',
    'embedded_1020', 'embedded_1021', 'embedded_1022', 'embedded_1023'

    ]]

    # Step 3: Feature Scaling
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    #print("Data after scaling:")
    #print(df.head(2))

    # Step 4: Finalizing Data for Modeling
    df_to_model = df.drop(columns=drop_columns)
    print("Final Data for Modeling Saved.")
    print(df_to_model.shape)

    # Optional: Save the prepared data to a new CSV file for easy access in future modeling.
    df_to_model.to_csv(output_path, index=False)
    
    return df_to_model

# Example usage:
#prepare_data_for_modeling(
#    filepath='../Data/Clean-Data/eda_data.csv',
#    drop_columns=['id', 'description', 'interactions_trans'],
#    numerical_features=['followers', 'num_posts', 'description_length'],
#    dummy_columns=['category', 'day_of_week', 'time_of_day']
#)



# Data Preprocessing for Model Training

def preprocess_data_for_modeling(data):
    """
    This function preprocesses the input dataset to prepare it for model training.
    It loads the data, converts data types, splits the data into training and testing sets,
    scales the features, and saves the preprocessed data to disk.

    Args:
    data (str): The file path to the cleaned dataset.

    Returns:
    None: This function saves the output directly to files.
    """


    # Step 1: Load the Cleaned Data
    # Load the final data from the previous notebook
    df = pd.read_csv(data)
    
    # Display basic information about the dataset
    #print(df.info())
    #print(df.head(2))
    
    # Change data types from bool to int for every column with a boolean type
    for col in df.columns:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)

    # Step 2: Define Features and Target Variable
    # Define the features (X) and the target variable (y)
    features = df.drop('interactions', axis=1)
    target = df['interactions']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=0)
    
    # Ensure all column names are strings (for compatibility with some models)
    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)

    # Step 3: Feature Scaling
    # Initialize the MinMaxScaler and fit it to the training data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Verify the scaled data
    print("Scaled Training Data:")
    print(X_train_scaled[:2])

    # Step 4: Save the Preprocessed Data
    # Save the preprocessed data for use in the next notebook
    pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv('../Data/Clean-Data/X_train_scaled.csv', index=False)
    pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv('../Data/Clean-Data/X_test_scaled.csv', index=False)
    y_train.to_csv('../Data/Clean-Data/y_train.csv', index=False)
    y_test.to_csv('../Data/Clean-Data/y_test.csv', index=False)
    
    print("Preprocessed data saved.")

# Example of how to call the function
# preprocess_data_for_modeling('../Data/Clean-Data/eda_data.csv')




def train_and_evaluate_models(X_train_scaled,X_test_scaled,y_train,y_test):
    """
    This function loads training and testing data, trains multiple regression models,
    and evaluates their performance on predicting Instagram post interactions.
    
    Parameters:
    - train_data_path: str, path to the training data CSV files.
    - test_data_path: str, path to the testing data CSV files.
    
    The function will print the performance metrics of each model.
    """
    
    # Load the preprocessed data
    X_train_scaled = pd.read_csv(X_train_scaled)
    X_test_scaled = pd.read_csv(X_test_scaled)
    y_train = pd.read_csv(y_train).values.ravel()
    y_test = pd.read_csv(y_test).values.ravel()
    

    # Model 1: XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=50,
        learning_rate=0.2,
        max_depth=3,
        reg_alpha=6.0,
        reg_lambda=10.0,
        random_state=42
    )
    xgb_model.fit(X_train_scaled, y_train)
    y_pred = xgb_model.predict(X_test_scaled)
    print_evaluation("XGBoost", y_test, y_pred)

    # Model 2: Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=50,
        max_depth=3,
        random_state=42
    )
    rf_model.fit(X_train_scaled, y_train)
    y_pred = rf_model.predict(X_test_scaled)
    print_evaluation("Random Forest", y_test, y_pred)

    # Model 3: Gradient Boosting
    gb_model = GradientBoostingRegressor(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    gb_model.fit(X_train_scaled, y_train)
    y_pred = gb_model.predict(X_test_scaled)
    print_evaluation("Gradient Boosting", y_test, y_pred)

    # Model 4: TensorFlow Neural Network
    tf_model = create_tf_model(X_train_scaled.shape[1])
    history = tf_model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
    y_pred = tf_model.predict(X_test_scaled).flatten()
    print_evaluation("TensorFlow Neural Network", y_test, y_pred)

def create_tf_model(input_dim):
    """Defines and compiles a TensorFlow neural network model."""
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def print_evaluation(model_name, y_true, y_pred):
    """Prints evaluation metrics for a given model."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} Model Performance:")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R-squared: {r2}\n")

# Example usage:
# train_and_evaluate_models('../Data/Clean-Data', '../Data/Clean-Data')




import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, make_scorer
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def hyperparameter_tuning_and_evaluation(X_train_scaled, X_test_scaled, y_train, y_test):
    """
    Perform hyperparameter tuning and evaluation on several regression models and identify the best model based on R-squared.
    Save the best model and the scaler used for feature scaling to disk.

    Parameters:
    - X_train_scaled: str, path to the CSV file for scaled training features.
    - X_test_scaled: str, path to the CSV file for scaled testing features.
    - y_train: str, path to the CSV file for training target values.
    - y_test: str, path to the CSV file for testing target values.

    Returns:
    - best_model_name: str, the name of the best performing model.
    - best_params: dict, the parameter set leading to the best performance.
    """
    # Load data
    X_train_scaled = pd.read_csv(X_train_scaled)
    X_test_scaled = pd.read_csv(X_test_scaled)
    y_train = pd.read_csv(y_train).values.ravel()
    y_test = pd.read_csv(y_test).values.ravel()

    # Define and tune models
    models = {
        'XGBoost': xgb.XGBRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }

    param_grids = {
        'XGBoost': {
            'n_estimators': [50],
            'max_depth': [3],
            'learning_rate': [0.2]
        },
        'Random Forest': {
            'n_estimators': [50],
            'max_depth': [5],
            'min_samples_split': [2]
        },
        'Gradient Boosting': {
            'n_estimators': [50],
            'learning_rate': [0.01],
            'min_samples_split': [2],
            'min_samples_leaf': [1]
        }
    }

    scoring = make_scorer(r2_score, greater_is_better=True)
    best_models = {}
    model_performance = {}

    for name, model in models.items():
        grid_search = GridSearchCV(model, param_grids[name], scoring=scoring, cv=5, verbose=1, n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)
        best_models[name] = grid_search.best_estimator_
        y_pred = best_models[name].predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        model_performance[name] = {'MAE': mae, 'MSE': mse, 'R2': r2, 'Model': best_models[name]}
        print(f"{name} \nModel with Best Parameters: \nMAE: {mae}, \nMSE: {mse}, \nR2: {r2}")

    # Determine the best model based on R-squared
    best_model_name = max(model_performance, key=lambda x: model_performance[x]['R2'])
    best_model = model_performance[best_model_name]['Model']
    best_params = best_model.get_params()

    # Save the scaler
    # Concatenate the features (X) data
    X_combined = pd.concat([X_train_scaled, X_test_scaled], axis=0).reset_index(drop=True)

    # Save the combined features
    X_combined.to_csv('../Data/Clean-Data/X_combined.csv', index=False)
    print("Combined features saved successfully!")

    # Initialize and fit the scaler
    scaler = MinMaxScaler()
    scaler.fit(X_combined)
    scaled_features = scaler.fit_transform(X_combined)
    scaler.transform(X_combined)

    # Save the scaler
    with open('scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)
    print("Scaler saved successfully!")

    # Save the best model
    with open('model.pkl', 'wb') as file:
        pickle.dump(best_model, file)
    print(f"Best model ({best_model_name}) saved as 'best_model.pkl'")

    # Verify that the model can be loaded
    with open('model.pkl', 'rb') as file:
        best_model = pickle.load(file)
    print("Model loaded successfully for verification!")
    print(f"The best model based on R-squared is: {best_model_name} with R2: {model_performance[best_model_name]['R2']}")
    print("Model and scaler saved successfully.")

    return best_model_name, best_params

# Example usage:
#X_train_scaled = 'path/to/X_train_scaled.csv'
#X_test_scaled = 'path/to/X_test_scaled.csv'
#y_train = 'path/to/y_train.csv'
#y_test = 'path/to/y_test.csv'
#best_model_name, best_params = hyperparameter_tuning_and_evaluation(X_train_scaled, X_test_scaled, y_train, y_test)



def model_evaluation_and_selection(X_train_scaled, X_test_scaled, y_train, y_test):
    """
    This function trains multiple regression models, evaluates their performance,
    selects the best model based on R-squared, and saves the best model and scaler.
    
    Parameters:
    - X_train_scaled: pd.DataFrame, scaled training features
    - X_test_scaled: pd.DataFrame, scaled testing features
    - y_train: pd.Series or np.array, training target values
    - y_test: pd.Series or np.array, testing target values
    
    Returns:
    - best_model_name: str, name of the best performing model
    - best_model: object, the best performing model
    """
      
    # Load the preprocessed data
    X_train_scaled = pd.read_csv(X_train_scaled)
    X_test_scaled = pd.read_csv(X_test_scaled)
    y_train = pd.read_csv(y_train).values.ravel()
    y_test = pd.read_csv(y_test).values.ravel()
    

    # Dictionary to store model performances
    model_performance = {}

    # Model 1: XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=50, learning_rate=0.2, max_depth=3, reg_alpha=6.0, reg_lambda=10.0, random_state=42)
    xgb_model.fit(X_train_scaled, y_train)
    y_pred = xgb_model.predict(X_test_scaled)
    model_performance['XGBoost'] = {'Model': xgb_model, 'R2': r2_score(y_test, y_pred)}
    print_evaluation("XGBoost", y_test, y_pred)

    # Model 2: Random Forest
    rf_model = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    y_pred = rf_model.predict(X_test_scaled)
    model_performance['Random Forest'] = {'Model': rf_model, 'R2': r2_score(y_test, y_pred)}
    print_evaluation("Random Forest", y_test, y_pred)

    # Model 3: Gradient Boosting
    gb_model = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
    gb_model.fit(X_train_scaled, y_train)
    y_pred = gb_model.predict(X_test_scaled)
    model_performance['Gradient Boosting'] = {'Model': gb_model, 'R2': r2_score(y_test, y_pred)}
    print_evaluation("Gradient Boosting", y_test, y_pred)

    # Model 4: TensorFlow Neural Network
    tf_model = create_tf_model(X_train_scaled.shape[1])
    history = tf_model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
    y_pred = tf_model.predict(X_test_scaled).flatten()
    model_performance['TensorFlow Neural Network'] = {'Model': tf_model, 'R2': r2_score(y_test, y_pred)}
    print_evaluation("TensorFlow Neural Network", y_test, y_pred)

    # Determine the best model based on R-squared
    best_model_name = max(model_performance, key=lambda x: model_performance[x]['R2'])
    best_model = model_performance[best_model_name]['Model']
    best_params = best_model.get_params() if hasattr(best_model, 'get_params') else {}
    model = best_model
    # Save the scaler
    X_combined = pd.concat([X_train_scaled, X_test_scaled], axis=0).reset_index(drop=True)
    X_combined.to_csv('X_combined.csv', index=False)
    #print("Combined features saved successfully!")

    scaler = MinMaxScaler()
    scaler.fit(X_combined)
    with open('scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)
    print("Scaler saved successfully!")

    # Save the best model
    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)
    print(f"Best model ({best_model_name}) saved as 'model.pkl'")

    # Verify that the model can be loaded
    with open('model.pkl', 'rb') as file:
          model = pickle.load(file)
    print("Model loaded successfully for verification!")
    print(f"The best model based on R-squared is: {best_model_name} with R2: {model_performance[best_model_name]['R2']}")
    print("Model and scaler saved successfully.")

    return best_model_name, best_model

def create_tf_model(input_dim):
    """Defines and compiles a TensorFlow neural network model."""
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def print_evaluation(model_name, y_true, y_pred):
    """Prints evaluation metrics for a given model."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} Model Performance:")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R-squared: {r2}\n")

# Example usage:
# best_model_name, best_model = model_evaluation_and_selection(X_train_scaled, X_test_scaled, y_train, y_test)