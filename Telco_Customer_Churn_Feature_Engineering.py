# Telco müşteri churn verileri, üçüncü çeyrekte Kaliforniya'daki 7043 müşteriye ev telefonu ve İnternet hizmetleri sağlayan
# hayali bir telekom şirketi hakkında bilgi içerir. Hangi müşterilerin hizmetlerinden ayrıldığını, kaldığını veya hizmete kaydolduğunu içermektedir.

# 21 Değişken 7043 Gözlem
# CustomerId : Müşteri İd’si
# Gender : Cinsiyet
# SeniorCitizen : Müşterinin yaşlı olup olmadığı (1, 0)
# Partner : Müşterinin bir ortağı olup olmadığı (Evet, Hayır) ? Evli olup olmama
# Dependents : Müşterinin bakmakla yükümlü olduğu kişiler olup olmadığı (Evet, Hayır) (Çocuk, anne, baba, büyükanne)
# tenure : Müşterinin şirkette kaldığı ay sayısı
# PhoneService : Müşterinin telefon hizmeti olup olmadığı (Evet, Hayır)
# MultipleLines : Müşterinin birden fazla hattı olup olmadığı (Evet, Hayır, Telefon hizmeti yok)
# InternetService : Müşterinin internet servis sağlayıcısı (DSL, Fiber optik, Hayır)
# OnlineSecurity : Müşterinin çevrimiçi güvenliğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# OnlineBackup : Müşterinin online yedeğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# DeviceProtection : Müşterinin cihaz korumasına sahip olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# TechSupport : Müşterinin teknik destek alıp almadığı (Evet, Hayır, İnternet hizmeti yok)
# StreamingTV : Müşterinin TV yayını olup olmadığı (Evet, Hayır, İnternet hizmeti yok) Müşterinin, bir üçüncü taraf sağlayıcıdan televizyon programları yayınlamak için İnternet hizmetini kullanıp kullanmadığını gösterir
# StreamingMovies : Müşterinin film akışı olup olmadığı (Evet, Hayır, İnternet hizmeti yok) Müşterinin bir üçüncü taraf sağlayıcıdan film akışı yapmak için İnternet hizmetini kullanıp kullanmadığını gösterir
# Contract : Müşterinin sözleşme süresi (Aydan aya, Bir yıl, İki yıl)
# PaperlessBilling : Müşterinin kağıtsız faturası olup olmadığı (Evet, Hayır)
# PaymentMethod : Müşterinin ödeme yöntemi (Elektronik çek, Posta çeki, Banka havalesi (otomatik), Kredi kartı (otomatik))
# MonthlyCharges : Müşteriden aylık olarak tahsil edilen tutar
# TotalCharges : Müşteriden tahsil edilen toplam tutar
# Churn : Müşterinin kullanıp kullanmadığı (Evet veya Hayır) - Geçen ay veya çeyreklik içerisinde ayrılan müşteriler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv("Telco_feature_eng/Telco-Customer-Churn.csv")
df.head()
df.shape
df.info()
df.isnull().sum()
# TotalCharges sayısal bir değişken olmalı
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce') # coerce seçimi eğer numerik değere çevirilemez ise o gözlemi NaN olarak ata demek için kullanıldı.
df.isnull().sum() # 11 tane gözlemin TotalCharges değeri NaN olarak atanmış.
df["TotalCharges"][df["TotalCharges"].isnull()] # Örneğin 488. gözlem Nan atanmış
df.loc[488]["TotalCharges"] # NaN girilen değelerin boş girildiğini görüyoruz.

df["Churn"] = df["Churn"].apply(lambda x: 1 if x=="Yes" else 0)

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col)
# Kategorik değişkenlerin oranına baktığımızda veri seti genel olarak dengeli dağılmış görünüyor.

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=True)

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Churn", col)

# Numerik değişkenlerin hedef değişkene göre analizini yaptığımızda tenure sayısının yüksek olmasının churn oranına bariz bir şekilde pozitif etki ettiğini
# TotalCharges değişkeninin yüksek olmasının da yüksek olmasının churn oranına pozitif etki ettiğini görebiliriz (totalcharges tenur ile de bağlantılı olduğu için böyle olması normaldir.)

def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)

# Gender değişkenine baktığımızda cinsiyet veri setinde eşit oranda dağılmış iki değişken içinde %26 bir churn oranı var. Yani cinsiyetin churn oranı üzerinde bariz bir etkisi yoktur diyebiliriz
# Partner değişkenine baktığımızda, bekar olanlar ve olmayanlar eşit dağılmış bekar olanlar %20 olmayanlar %30 churn oranına sahip. Yani evlilik durumu churn oranı üzerinde azda olsa etkiye sahip görünmekte
# Dependents değişkenine baktığımızda. Yes değerini alan kısım veri setinin %30 unu oluşturmakta değerlendirmeye almak için yeterli bir miktar, dependents değişkeni yes olan bireylerin churn oranı daha az oluyor yorumunu yapabiliriz
# PhoneService, telefon servisi olmayanlar veri setinin çok az bir kısmını oluşturuyor olsalar da telefon servisinin olup olmaması churn oranında etki etmiyor gibi gözükmekte
# MultipleLines, Churn oranına etki etmediği yorumu yapılabilir
# InternetService, Hiç internet servisi almayanların bariz şekilde churn oranları düşük, firmanın internet servisinin kötü olduğu yorumu yapılabilir. Ayrıca DSL yerine fiber optic internet servisi
# kullananların churn oranı çok daha yüksek Fiber optic servislerinin memnuniyet oranı çok düşük internet servisi tarafından yaşanacak churnlerin çoğunluğuna sebep olmakta bunun üzerine çalışma
# yapılabilir
# OnlineSecurity, güvenlik hizmeti almayanların churn oranı bariz şekilde daha fazla, müşterilerin güvenlik hizmeti almama nedenleri araştırılıp gerekli kampanyalarla hizmet almaya teşvik edilebilir,
# churn oranına oldukça etkisi olacaktır. Ayrıca Fiber optic kullanıcıların miktarı ve churn oranı ile güvenlik  hizmeti almayanlarınki birbirine çok benziyor fiber optic servis kullananlara güvenlik
# hizmeti şirket tarafından verilmiyor olabilir.
# OnlineBackup, online backup servisi almayanların churn oranı daha fazla yine bunun için ayrıca analiz yapılıp servisi almaya teşvik edilebilir.
# DeviceProtection, Cihaz koruma hizmeti almayanların yine churn oranı bariz fazladır, önceki hizmetler için uygulanan strajiler uygulanabilir.
# TechSupport, yine aynı şekilde destek almayanların churn oranı fazla
# StreamingTV, Tv yayın hizmetinin churn oranlarında bir etkisi yoktur yorumu yapılabilir
# Contract, Aylık kontrat yapanların churn oranı daha fazla
# PaperlessBilling, elektronik fatura isteyen müşterilerin churn oranı daha fazla, genç insanların elektronik faturayı daha çok tercih edeceğini düşünerek, gençlerin firmanın hizmetinden yaşça daha olgun
# insanlara göre daha az memnun olduğu yorumu yapılabilir.
# PaymentMethod, elektronik çek yöntemini kullananların churn oranı bariz fazladır.
# SeniorCitizen, Yaşlu müşterilerin oranı az da olsa churn oranlarının bariz yüksek olduğu yorumu yapılabilir.

df[num_cols].corr()

# Korelasyon Matrisi
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show(block=True)

df.corrwith(df["Churn"])

df.isnull().sum() # TotalCharges değişkeninde 11 adet null değer vardır.
df.isnull().sum()/df.shape[0] # Çok çok düşük bir oranda null değer var drop edilebilir veya aylık ödeme ile tenur oranı çarpılarak doldurulabilir.
df.loc[df["TotalCharges"].isnull(), "TotalCharges"] = df.loc[df["TotalCharges"].isnull(), "MonthlyCharges"] * df.loc[df["TotalCharges"].isnull(), "tenure"]
# df.loc[df["TotalCharges"].isnull(), "tenure"] incelendiğinde null değere sahip olan bireylerin aylık ödemesi görünmekte ancak tenur değerleri 0'dır. Yani bu müşteriler daha 1. ayını bile doldurmamış

dff = df.copy()
cat_cols = [col for col in cat_cols if col not in ["Churn"]]
cat_cols

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
dff = one_hot_encoder(dff, cat_cols, drop_first=True)


y = dff["Churn"]
X = dff.drop(["Churn","customerID"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

catboost_model = CatBoostClassifier(verbose=False, random_state=12345).fit(X_train, y_train)
y_pred = catboost_model.predict(X_test)


print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 4)}")
print(f"Recall: {round(recall_score(y_pred,y_test),4)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 4)}")
print(f"F1: {round(f1_score(y_pred,y_test), 4)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 4)}")


def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# Aykırı Değer Analizi ve Baskılama İşlemi
for col in num_cols:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

df.loc[(df["tenure"]>=0) & (df["tenure"]<=12),"NEW_TENURE_YEAR"] = "NewCustomer"
df.loc[(df["tenure"]>12) & (df["tenure"]<=36),"NEW_TENURE_YEAR"] = "OrdinaryCustomer"
df.loc[(df["tenure"]>36) & (df["tenure"]<=72),"NEW_TENURE_YEAR"] = "OldCustomer"
df.drop("tenure", axis=1, inplace=True)

# Herhangi bir destek, yedek veya koruma almayan kişiler
df["NEW_noProt"] = df.apply(lambda x: 1 if (x["OnlineBackup"] != "Yes") or (x["DeviceProtection"] != "Yes") or (x["TechSupport"] != "Yes") else 0, axis=1)

# Kişinin toplam aldığı servis sayısı
df['NEW_TotalServices'] = (df[['PhoneService', 'InternetService', 'OnlineSecurity',
                                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                       'StreamingTV', 'StreamingMovies']]== 'Yes').sum(axis=1)

# Herhangi bir streaming hizmeti alan kişiler
df["NEW_FLAG_ANY_STREAMING"] = df.apply(lambda x: 1 if (x["StreamingTV"] == "Yes") or (x["StreamingMovies"] == "Yes") else 0, axis=1)

# Kişi otomatik ödeme yapıyor mu?
df["NEW_FLAG_AutoPayment"] = df["PaymentMethod"].apply(lambda x: 1 if x in ["Bank transfer (automatic)","Credit card (automatic)"] else 0)

cat_cols, num_cols, cat_but_car = grab_col_names(df)
one_hot_columns =[col for col in cat_cols if col not in ["Churn","customerID"]]

df = pd.get_dummies(df, columns=one_hot_columns, drop_first=True)
df.nunique()
df.shape
df.head()
df.drop("customerID", axis=1, inplace=True)

scaler=StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

y = df["Churn"]
X = df.drop(["Churn"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

catboost_model = CatBoostClassifier(verbose=False, random_state=12345).fit(X_train, y_train)
y_pred = catboost_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 4)}")
print(f"Recall: {round(recall_score(y_pred,y_test),4)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 4)}")
print(f"F1: {round(f1_score(y_pred,y_test), 4)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 4)}")

"""
Accuracy: 0.7922
Recall: 0.6517
Precision: 0.5052
F1: 0.5692
Auc: 0.7407
"""