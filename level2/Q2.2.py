import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures


df=pd.read_csv("w2_position_salaries-220925-153437.csv")
df.head()



from matplotlib import pyplot as plt
plt.figure(figsize=(6,4))
plt.plot(df['Level'],df['Salary'],'ro')
plt.title('Salary vs Level')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()


X = df.iloc[:, 1:2].values  
y = df.iloc[:, 2].values 
#degree 1-linear Regression

lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin_reg = lin_reg.predict(X)
r2_score_lin_reg = r2_score(y, y_pred_lin_reg)

print(f"R² Score (Linear Regression): {r2_score_lin_reg}")



#degree 2-Polinomial
poly_reg_2 = PolynomialFeatures(degree=2)
X_poly_2 = poly_reg_2.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly_2, y)
y_pred_poly_2 = lin_reg_2.predict(X_poly_2)
r2_score_poly_2 = r2_score(y, y_pred_poly_2)

print(f"R² Score (Polynomial Regression, Degree 2): {r2_score_poly_2}")



#degree 3-Polinomial
poly_reg_3 = PolynomialFeatures(degree=3)
X_poly_3 = poly_reg_3.fit_transform(X)
lin_reg_3 = LinearRegression()
lin_reg_3.fit(X_poly_3, y)
y_pred_poly_3 = lin_reg_3.predict(X_poly_3)
r2_score_poly_3 = r2_score(y, y_pred_poly_3)

print(f"R² Score (Polynomial Regression, Degree 3): {r2_score_poly_3}")



#degree 4-Polinomial

poly_reg_4 = PolynomialFeatures(degree=4)
X_poly_4 = poly_reg_4.fit_transform(X)
lin_reg_4 = LinearRegression()
lin_reg_4.fit(X_poly_4, y)
y_pred_poly_4 = lin_reg_4.predict(X_poly_4)
r2_score_poly_4 = r2_score(y, y_pred_poly_4)

print(f"R² Score (Polynomial Regression, Degree 4): {r2_score_poly_4}")





