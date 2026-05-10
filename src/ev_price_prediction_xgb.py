

# ====================import pandas as pd
train=pd.read_csv("C:/Users/user/Desktop/open/train.csv")
test=pd.read_csv("C:/Users/user/Desktop/open/test.csv")

train.info()
test.info()

train1=train.copy()
test1=test.copy()

train1

def create_improved_features(data):
    
    data['제조사_모델_상태'] = data['제조사'] + '_' + data['모델']+ '_' + data['차량상태']
    
    data['주행거리비율'] = data.apply(
        lambda row: row['주행거리(km)'] if row['연식(년)'] == 0 else row['주행거리(km)'] / row['연식(년)'],
        axis=1)
    
    current_year = 2025 
    data['사용연한주행거리비율'] = data['주행거리(km)'] / (current_year - data['연식(년)'] + 1)

    data['전비(km/kWh)'] = data['주행거리(km)'] / data['배터리용량']
    
    data['중고여부'] = data['차량상태'].apply(lambda x: 0 if x == 'Brand New' else 1)
    
    drive_weight = {'AWD': 1.2, 'RWD': 1.1, 'FWD': 1.0}
    data['구동방식가중치'] = data['구동방식'].map(drive_weight)
    
    data['사고가중치'] = data['사고이력'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    return data

# train_data와 test_data에 적용
train1 = create_improved_features(train1)
test1 = create_improved_features(test1)

# 생성된 데이터 확인
print(train1[['제조사_모델_상태','주행거리비율','사용연한주행거리비율', '전비(km/kWh)', '중고여부','구동방식가중치', '사고가중치']])
print(test1[['제조사_모델_상태','주행거리비율','사용연한주행거리비율', '전비(km/kWh)', '중고여부','구동방식가중치', '사고가중치']])


import pandas as pd
import numpy as np
from scipy import stats

features = ['제조사', '모델', '차량상태','제조사_모델_상태', '구동방식', '사고이력']
target = '배터리용량'

original_train1 = train1.copy()

if train1[target].isnull().any():
    train1 = train1.dropna(subset=[target])

anova_results = {}
for feature in features:
    if feature in train1.columns:
        group_data = [train1[target][train1[feature] == category] for category in train1[feature].unique()]
        f_stat, p_value = stats.f_oneway(*group_data)
        anova_results[feature] = (f_stat, p_value)

print("ANOVA 결과 (F-statistic, p-value):")
for feature, (f_stat, p_value) in anova_results.items():
    print(f"{feature}: F-statistic = {f_stat}, p-value = {p_value}")

significant_features = [feature for feature, (f_stat, p_value) in anova_results.items() if p_value < 0.05]
print("유의미한 변수들:", significant_features)

train1 = original_train1.copy()


import pandas as pd
import numpy as np
from scipy.stats import pearsonr

features = ['보증기간(년)', '연식(년)', '주행거리(km)','주행거리비율','사용연한주행거리비율', '전비(km/kWh)', '중고여부','구동방식가중치', '사고가중치']
target = '배터리용량'

original_train1 = train1.copy()

train1_clean = train1.dropna(subset=[target] + features)

results = {}
for feature in features:
    corr, p_value = pearsonr(train1_clean[feature], train1_clean[target])
    results[feature] = {'correlation': corr, 'p-value': p_value}

for feature, values in results.items():
    print(f"{feature}: 상관계수 = {values['correlation']:.4f}, p-value = {values['p-value']:.4f}")

significant_results = [feature for feature, values in results.items() if values['p-value'] < 0.05]
print("유의미한 변수들:", significant_results)

train1 = original_train1.copy()


import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

features = ['제조사', '모델', '차량상태', '구동방식', '제조사_모델_상태', '주행거리(km)', '보증기간(년)',
            '주행거리비율', '중고여부', '구동방식가중치']
target = '배터리용량'

data_train = train1[train1[target].notnull()]
data_missing = train1[train1[target].isnull()]

X_train = data_train[features]
y_train = data_train[target]
X_missing = data_missing[features]

categorical_cols = ['제조사', '모델', '차량상태', '구동방식', '제조사_모델_상태']
numerical_cols = ['주행거리(km)', '보증기간(년)', '주행거리비율', '중고여부', '구동방식가중치']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),  # 결측치 평균 대체
            ('scaler', StandardScaler())  # 스케일링
        ]), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)  # 원-핫 인코딩
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor())
])

model.fit(X_train, y_train)

predicted_values = model.predict(X_missing)

train1.loc[train1[target].isnull(), target] = predicted_values

train1['사용연한주행거리비율'] = train1['주행거리(km)'] / (train1['배터리용량'])
train1['전비(km/kWh)'] = train1['주행거리(km)'] / (train1['배터리용량'])

print(train1.info())


import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

features = ['제조사', '모델', '차량상태', '구동방식', '제조사_모델_상태', '주행거리(km)', '보증기간(년)',
            '주행거리비율', '중고여부', '구동방식가중치']
target = '배터리용량'

data_train = test1[test1[target].notnull()]
data_missing = test1[test1[target].isnull()]

X_train = data_train[features]
y_train = data_train[target]
X_missing = data_missing[features]

categorical_cols = ['제조사', '모델', '차량상태', '구동방식', '제조사_모델_상태']
numerical_cols = ['주행거리(km)', '보증기간(년)', '주행거리비율', '중고여부', '구동방식가중치']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[  
            ('scaler', StandardScaler()) 
        ]), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)  # 원-핫 인코딩
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor())
])

model.fit(X_train, y_train)

predicted_values = model.predict(X_missing)

test1.loc[test1[target].isnull(), target] = predicted_values

test1['사용연한주행거리비율'] = test1['주행거리(km)'] / (test1['배터리용량'])
test1['전비(km/kWh)'] = test1['주행거리(km)'] / (test1['배터리용량']) 

print(test1.info())


import pandas as pd
import numpy as np
from scipy import stats

features = ['제조사', '모델', '차량상태','제조사_모델_상태', '구동방식', '사고이력']
target = '가격(백만원)'

original_train1 = train1.copy()

if train1[target].isnull().any():
    train1 = train1.dropna(subset=[target])

anova_results = {}
for feature in features:
    if feature in train1.columns:
        group_data = [train1[target][train1[feature] == category] for category in train1[feature].unique()]
        f_stat, p_value = stats.f_oneway(*group_data)
        anova_results[feature] = (f_stat, p_value)

print("ANOVA 결과 (F-statistic, p-value):")
for feature, (f_stat, p_value) in anova_results.items():
    print(f"{feature}: F-statistic = {f_stat}, p-value = {p_value}")

significant_features = [feature for feature, (f_stat, p_value) in anova_results.items() if p_value < 0.05]
print("유의미한 변수들:", significant_features)

train1 = original_train1.copy()


import pandas as pd
import numpy as np
from scipy.stats import pearsonr

features = ['보증기간(년)', '연식(년)','배터리용량','주행거리(km)','주행거리비율','사용연한주행거리비율', '전비(km/kWh)', '중고여부','구동방식가중치', '사고가중치']
target = '가격(백만원)'

original_train1 = train1.copy()

train1_clean = train1.dropna(subset=[target] + features)

results = {}
for feature in features:
    corr, p_value = pearsonr(train1_clean[feature], train1_clean[target])
    results[feature] = {'correlation': corr, 'p-value': p_value}

for feature, values in results.items():
    print(f"{feature}: 상관계수 = {values['correlation']:.4f}, p-value = {values['p-value']:.4f}")

significant_results = [feature for feature, values in results.items() if values['p-value'] < 0.05]
print("유의미한 변수들:", significant_results)

train1 = original_train1.copy()


import pandas as pd
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

features = ['제조사', '모델', '차량상태', '구동방식','제조사_모델_상태', 
            '보증기간(년)', '연식(년)', '배터리용량', '주행거리(km)', '주행거리비율',
            '사용연한주행거리비율', '전비(km/kWh)', '중고여부', '구동방식가중치']
target = '가격(백만원)'

X_train = train1[features]
y_train = train1[target]

X_test = test1[features]

categorical_cols = ['제조사', '모델', '차량상태', '구동방식','제조사_모델_상태','보증기간(년)', '연식(년)']
numerical_cols = ['배터리용량', '주행거리(km)','주행거리비율', '사용연한주행거리비율', '전비(km/kWh)', '중고여부', '구동방식가중치']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('scaler', StandardScaler()) 
        ]), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(objective='reg:squarederror', random_state=42))
])

model.fit(X_train, y_train)

predicted_prices = model.predict(X_test)

test1[target] = predicted_prices

result = test1[['ID', '가격(백만원)']]
print(result)

result.to_csv('C:/Users/user/Desktop/open/sample_submission.csv', index=False)
