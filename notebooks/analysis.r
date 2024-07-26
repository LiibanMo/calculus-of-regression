library(fastDummies)

# Insert desired file path
file_path <- "/Users/liibanmohamud/Downloads/Machine Learning/Practice/MLOps/Advanced Multiple Linear Regression/data/Student_Performance.csv"

# Read desired .csv file
df <- read.csv(file_path)

# -------------------------- Preprocessing -------------------------- #

# One-hot encoding
df_to_one_hot <- df[sapply(df, is.character)]
columns_to_one_hot <- colnames(df_to_one_hot)
df_one_hot <- dummy_cols(df, select_columns = columns_to_one_hot, remove_first_dummy = TRUE,remove_selected_columns = TRUE)
df_numeric_columns <- df_one_hot[sapply(df_one_hot, is.numeric)]
df_numeric_columns_scaled <- scale(df_numeric_columns)
data <- as.data.frame(df_numeric_columns_scaled) # The final preprocessed dataframe

# -------------------------- Fitting model -------------------------- #

y <- data$Performance.Index # Change to correspond with your data
model <- lm(y ~ ., data = data)
summary(model)