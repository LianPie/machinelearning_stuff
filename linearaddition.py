# Polynomial Regression
poly_features = PolynomialFeatures(degree=2)
X_train_quadratic = poly_features.fit_transform(X_train)
quadratic = LinearRegression()
quadratic.fit(X_train_quadratic, y_train)
y_test_predicted = quadratic.predict(poly_features.transform(X_test))
metrics.r2_score(y_test, y_test_predicted)

# ==============================
# 6. Logistic Regression
# ==============================
from sklearn.linear_model import LogisticRegression
iris = pd.read_csv('Iris.csv')
iris.drop("Id", axis=1, inplace=True)
X = iris.drop(['Species'], axis=1)
y = iris['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
pd.crosstab(logreg.predict(X), y)

# ==============================
# 7. Data Preprocessing
# ==============================
df = pd.read_csv('Data.csv')

# Handling missing values
df_dropna = df.dropna()  # Drop rows with NaN
df_fillna = df.fillna(df.mean())  # Fill NaN with mean

# Encoding categorical variables
df_encoded = pd.get_dummies(df, columns=['Country'])  # One-hot encoding
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Using sklearn
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
le = LabelEncoder()
y = le.fit_transform(y)

# Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler, StandardScaler
mm = MinMaxScaler()
X_train[:, 3:] = mm.fit_transform(X_train[:, 3:])
X_test[:, 3:] = mm.transform(X_test[:, 3:])
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])