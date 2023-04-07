import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import seaborn as sns
from fitter import Fitter, get_common_distributions, get_distributions
from scipy.stats import gamma,lognorm,beta,expon,norm,iqr, scoreatpercentile
from scipy.optimize import minimize_scalar
from sklearn.svm import OneClassSVM
import random
import re
from scipy.stats import norm, lognorm, weibull_min, expon, gamma, uniform, kstest

class OutlierRemover_general_iqr(BaseEstimator,TransformerMixin): # our own class to remove outliers - we will insert it to the pipeline 
    def __init__(self,cols=[],factor=1.5):
        self.factor = factor # higher the factor, extreme would be the outliers removed.
        self.cols = cols
        
    def outlier_detector(self,X,y=None):
        X = pd.Series(X).copy()
        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)
        iqr = q3 - q1
        self.lower_bound.append(q1 - (self.factor * iqr)) 
        self.upper_bound.append(q3 + (self.factor * iqr))
        self.median.append(X.median()) # try to change

    def fit(self,X,y=None): # for each coulmn we will append corresponding boundary and the median value
        self.median = []
        self.lower_bound = []
        self.upper_bound = []
        X.apply(self.outlier_detector)
        return self
    
    def transform(self,X,y=None): # then, with transform we will check is a value goes beyond the boundary, if so we replace it
        X = pd.DataFrame(X).copy()
        for i in range(X.shape[1]):
            if X.columns[i] in self.cols:
                continue
            x = X.iloc[:, i].copy() # change the copy
            x[(x < self.lower_bound[i]) | (x > self.upper_bound[i])] = self.median[i] # replace outliers with the median
            X.iloc[:, i] = x # make the column copy

        return X # our transformed df
    
class Outliers_removal_ml_dif_rand(BaseEstimator,TransformerMixin): # working class
    def __init__(self,cols):
        self.data = {}
        self.scaler = StandardScaler()
        self.cols = cols
    
    def fit(self,X,y=None):
        length = len(X.columns)
        cur_perc = 0
        
        for col in X.columns:  # for each col
            d = X[[col]]
            d = self.scaler.fit_transform(d)            
            model = OneClassSVM(kernel='rbf', gamma='auto')#OneClassSVM(kernel= "linear") # train binary SVM 
            model.fit(d)
            outliers = model.predict(d) == -1 # the outliers will be -1 else 1, so bool vector True is outlier else False
            self.data[col] = outliers
            cur_perc += 1
            print(round(cur_perc/length, 2)*100)
            print()
        return self
    
    def transform(self,X,y=None):
        X = pd.DataFrame(X).copy().reset_index(drop=True) 
        for col in X.columns:
            x = X[col].copy()
            outlier_inds = self.data[col]  
            l = sum(outlier_inds)  
            h = X[col].tolist() # data for column
            f = Fitter(h, # try these 5 distrs
           distributions=['gamma', 
                          'lognorm',
                          "expon",
                          "norm"]) 
            f.fit()          
            try:
                d = f.get_best(method = 'sumsquare_error') 
                best_distr = list(d.keys())[0]
                params = d[best_distr] 
                l = sum(outlier_inds)        
                if best_distr == "gamma":
                    a_gamma, scale_gamma = params["a"],params["scale"]
                    x[outlier_inds] =  np.random.gamma(a_gamma, scale=scale_gamma, size=l) 
                elif best_distr == "lognorm":
                    s_lognorm, scale_lognorm = params["loc"],params["scale"]
                    x[outlier_inds] = np.random.lognormal(s_lognorm, scale_lognorm, l)       
                elif best_distr == "expon": 
                    scale = params["scale"]
                    x[outlier_inds] = np.random.exponential(scale, l)  
                elif best_distr == "norm": 
                    mu_norm, std_norm = params["loc"],params["scale"]
                    x[outlier_inds] = np.random.normal(mu_norm, std_norm, l)
            except KeyError: 
                x[outlier_inds]= x.median()

            X[col] = x
        return X
    
class OutlierRemover_distrs(BaseEstimator,TransformerMixin): 
    def __init__(self):
        self.fitters = {}
    def cal_bounds(self,best_distr,data,params): # get the bunds for different distrs
        if best_distr == "gamma":
            shape, loc, scale = params["shape"],params["loc"],params["scale"]
            theoretical_quantiles = gamma.ppf(np.linspace(0.01, 0.99, 99), shape, loc, scale)
            observed_quantiles = np.percentile(data, np.linspace(1, 99, 99))
            differences = np.abs(theoretical_quantiles - observed_quantiles)
            iqr_differences = iqr(differences)
            q1 = scoreatpercentile(data, 25)
            q3 = scoreatpercentile(data, 75)
            lower_bound = gamma.ppf(0.25 - 1.5*iqr_differences, shape, loc, scale)
            upper_bound = gamma.ppf(0.75 + 1.5*iqr_differences, shape, loc, scale)
            
        elif best_distr == "lognorm":
            s, loc, scale = params["s"],params["loc"],params["scale"]
            theoretical_quantiles = lognorm.ppf(np.linspace(0.01, 0.99, 99), s, loc, scale)
            observed_quantiles = np.percentile(data, np.linspace(1, 99, 99))
            differences = np.abs(theoretical_quantiles - observed_quantiles)
            iqr_differences = iqr(differences)
            q1 = scoreatpercentile(data, 25)
            q3 = scoreatpercentile(data, 75)
            lower_bound = lognorm.ppf(0.25 - 1.5*iqr_differences, s, loc, scale)
            upper_bound = lognorm.ppf(0.75 + 1.5*iqr_differences, s, loc, scale)
            
        elif best_distr == "beta":
            a, b, loc, scale = params["a"],params["b"],params["loc"],params["scale"]
            theoretical_quantiles = beta.ppf(np.linspace(0.01, 0.99, 99), a, b, loc, scale)
            observed_quantiles = np.percentile(data, np.linspace(1, 99, 99))
            differences = np.abs(theoretical_quantiles - observed_quantiles)
            iqr_differences = iqr(differences)
            q1 = scoreatpercentile(data, 25)
            q3 = scoreatpercentile(data, 75)
            lower_bound = beta.ppf(0.25 - 1.5*iqr_differences, a, b, loc, scale)
            upper_bound = beta.ppf(0.75 + 1.5*iqr_differences, a, b, loc, scale)
            
        elif best_distr == "expon":
            loc, scale = params["loc"], params["scale"]
            theoretical_quantiles = expon.ppf(np.linspace(0.01, 0.99, 99), loc, scale)
            observed_quantiles = np.percentile(data, np.linspace(1, 99, 99))
            differences = np.abs(theoretical_quantiles - observed_quantiles)
            iqr_differences = iqr(differences)
            q1 = scoreatpercentile(data, 25)
            q3 = scoreatpercentile(data, 75)
            lower_bound = expon.ppf(0.25 - 1.5*iqr_differences, loc, scale)
            upper_bound = expon.ppf(0.75 + 1.5*iqr_differences, loc, scale)
            
        else: # norm
            loc, scale = params["shape"],params["loc"],params["scale"]
            theoretical_quantiles = norm.ppf(np.linspace(0.01, 0.99, 99), loc, scale)
            observed_quantiles = np.percentile(data, np.linspace(1, 99, 99))
            differences = np.abs(theoretical_quantiles - observed_quantiles)
            iqr_differences = iqr(differences)
            q1 = scoreatpercentile(data, 25)
            q3 = scoreatpercentile(data, 75)
            lower_bound = norm.ppf(0.25 - 1.5*iqr_differences, loc, scale)
            upper_bound = norm.ppf(0.75 + 1.5*iqr_differences, loc, scale)       
        return lower_bound,upper_bound
        
    def fit(self,X,y=None):
        for col in X.columns:
            h = X[col].tolist()
            f = Fitter(h, # try these 5
           distributions=['gamma', # gamma
                          'lognorm', # lognormal
                          "beta", # beta
                          "expon", # exp
                          "norm"]) # gauss
            f.fit()
            self.fitters[col] = f         
        return self
    
    @staticmethod
    def neg_log_likelihood(lam, data): # max likelyhood method
        n = len(data)
        log_likelihood = n * np.log(lam) - lam * np.sum(data)
        return -log_likelihood

    def transform(self,X,y=None):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            f = self.fitters[col]
            data = X[col].tolist()
            try:
                d = f.get_best(method = 'sumsquare_error')
                best_distr = list(d.keys())[0] # norm
                params = d[best_distr] # params of distr
                lower_bound,upper_bound = self.cal_bounds(best_distr,data,params) # get the upper and lower bound for each distr
                x = X[col].copy() 
                l = len(x[(x < lower_bound) | (x > upper_bound)])          
                if best_distr == "gamma":
                    alpha, beta = params["shape"],params["scale"]
                    x[(x < lower_bound) | (x > upper_bound)] = [random.gammavariate(alpha, beta) for i in range(l)]
                elif best_distr == "lognorm":
                    mu, sigma = params["loc"],params["scale"]
                    x[(x < lower_bound) | (x > upper_bound)] = [random.lognormvariate(mu, sigma) for i in range(l)]            
                elif best_distr == "beta":
                    alpha, beta = params["shape"],params["scale"]
                    x[(x < lower_bound) | (x > upper_bound)] = [random.betavariate(alpha, beta) for i in range(l)]          
                elif best_distr == "expon":
                    res = minimize_scalar(self.neg_log_likelihood, args=(data,)) #OutlierRemover_distrs (use max likeluhhod method to estimate the lambda since mean could be 0 and lambda = 1/mean)
                    lambda_exp = res.x
                    x[(x < lower_bound) | (x > upper_bound)] = [random.expovariate(lambda_exp) for i in range(l)]            
                elif best_distr == "norm": 
                    mu, sigma = params["loc"],params["scale"]
                    x[(x < lower_bound) | (x > upper_bound)] = [random.gauss(mu, sigma) for i in range(l)]   
            except KeyError:
                x[(x < lower_bound) | (x > upper_bound)] = x.median() # if all distr were dropped just use the median
            X[col] = x 
        return X
"""  
class Outliers_removal_ml_dif_rand(BaseEstimator,TransformerMixin): # not working class
    def __init__(self):
        self.data = {}
        self.scaler = StandardScaler()
    
    def fit(self,X,y=None):
        length = len(X.columns)
        cur_perc = 0
        
        for col in X.columns:  # for each col
            d = X[[col]]
            d = self.scaler.fit_transform(d)
            model = OneClassSVM(kernel='rbf', gamma='auto')#OneClassSVM(kernel= "linear") # train binary SVM 
            model.fit(d)
            outliers = model.predict(d) == -1 # the outliers will be -1 else 1, so bool vector True is outlier else False
            self.data[col] = outliers
            cur_perc += 1
            print(round(cur_perc/length, 2)*100)
        return self
    
    def transform(self,X,y=None):
        print("!!!")
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            x = X[col].copy()
            data  = X[col]
            outlier_inds = self.data[col]  
            l = sum(outlier_inds)  
            
            mu_norm, std_norm = norm.fit(data)
            s_lognorm, loc_lognorm, scale_lognorm = lognorm.fit(data)
            c_weibull, loc_weibull, scale_weibull = weibull_min.fit(data)
            loc_expon, lambda_expon = expon.fit(data)
            a_gamma, loc_gamma, scale_gamma = gamma.fit(data)
            loc_unif, scale_unif = uniform.fit(data)

            ks_norm = kstest(data, norm(mu_norm, std_norm).cdf)[1]
            ks_lognorm = kstest(data, lognorm(s_lognorm, loc_lognorm, scale_lognorm).cdf)[1]
            ks_weibull = kstest(data, weibull_min(c_weibull, loc_weibull, scale_weibull).cdf)[1]
            ks_expon = kstest(data, expon(loc_expon, lambda_expon).cdf)[1]
            ks_gamma = kstest(data, gamma(a_gamma, loc_gamma, scale_gamma).cdf)[1]
            ks_unif = kstest(data, uniform(loc_unif, scale_unif).cdf)[1]

            p_values = [ks_norm, ks_lognorm, ks_weibull, ks_expon, ks_gamma, ks_unif]
            distr_names = ['normal', 'lognormal', 'weibull', 'exponential', 'gamma', 'uniform']
            best_fit = distr_names[np.argmin(p_values)]
            if best_fit == 'normal':
                rands = np.random.normal(mu_norm, std_norm, l)
            elif best_fit == "lognormal":
                rands =  np.random.lognormal(s_lognorm, scale_lognorm, l)
            elif best_fit == "weibull":
                rands = np.random.weibull(c_weibull, l) * scale_weibull
            elif best_fit == "exponential":
                rands = np.random.exponential(lambda_expon, l)
            elif best_fit == "gamma":
                rands = np.random.gamma(a_gamma, scale=scale_gamma, size=l)
            elif best_fit == "uniform":
                rands = np.random.uniform(loc_unif, scale_unif, l)
                
            x[outlier_inds] = rands 
            X[col] = x
        return X.reset_index(drop=True)
    
"""

class Outliers_removal_ml(BaseEstimator,TransformerMixin): # working class
    def __init__(self):
        self.data = {}
    
    def fit(self,X,y=None):
        print("!")
        length = len(X.columns)
        cur_perc = 0
        for col in X.columns:  # for each col
            d = X[[col]]
            model = OneClassSVM(kernel="linear") #OneClassSVM(kernel='rbf', gamma='auto') # train binary SVM 
            model.fit(d)
            outliers = model.predict(d) == -1 # the outliers will be -1 else 1, so bool vector True is outlier else False
            self.data[col] = outliers
            cur_perc += 1
            print(round(cur_perc/length, 2)*100)
        print("!!")
        return self
    
    @staticmethod
    def neg_log_likelihood(lam, data): # max likelyhood method
        n = len(data)
        log_likelihood = n * np.log(lam) - lam * np.sum(data)
        return -log_likelihood

    def transform(self,X,y=None):
        print("!!!")
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            print(col)
            outlier_inds = self.data[col]           
            x = X[col].copy()
            h = X[col].tolist() # data for column
            f = Fitter(h, # try these 5 distrs
           distributions=['gamma', # gamma
                          'lognorm', # lognormal
                          "beta", # beta
                          "expon", # exp
                          "norm"]) # gauss
            f.fit()
            #f.summary() see the graph and results
            try:
                d = f.get_best(method = 'sumsquare_error') # get the best one
                best_distr = list(d.keys())[0] # get its params
                params = d[best_distr] 
                l = sum(outlier_inds)       # whole num of outliers   
                print(l)
                if best_distr == "gamma": # for differetn distrs
                    alpha, beta = params["shape"],params["scale"] # get neccassary params
                    x[outlier_inds] =  list(random.gammavariate(alpha, beta) for i in range(l)) # generate nums (note that they are in the form generators and then we convrete them to list - saves a bit more time)
                elif best_distr == "lognorm":
                    mu, sigma = params["loc"],params["scale"]
                    x[outlier_inds] = list(random.lognormvariate(mu, sigma) for i in range(l))        
                elif best_distr == "beta":
                    alpha, beta = params["shape"],params["scale"]
                    x[outlier_inds] = list(random.betavariate(alpha, beta) for i in range(l))      
                elif best_distr == "expon": # here sit is a bit different since lambda can be 1/0 since mean could be 0 and lambda = 1/mean
                    res = minimize_scalar(self.neg_log_likelihood, args=(h,)) #OutlierRemover_distrs (use max likeluhhod method to estimate the lambda since mean could be 0 and lambda = 1/mean)
                    lambda_exp = res.x
                    x[outlier_inds] = list(random.expovariate(lambda_exp) for i in range(l))    
                elif best_distr == "norm": 
                    mu, sigma = params["loc"],params["scale"]
                    x[outlier_inds] = list(random.gauss(mu, sigma) for i in range(l)) 
            except KeyError: # in case non of the distr were fitted for some reason
                x[outlier_inds]= x.median() # just use the median
            X[col] = x
        print("!!!!")
        return X.reset_index(drop=True)
            
             
class Custom_Cat_Imputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self,X,y=None):
        X = pd.DataFrame(X).copy() # make the copy
        for col in X.columns: # for each column
            X[col] = X[col].interpolate(method='pad', limit_direction = "forward") # fffil
            X[col] = X[col].interpolate(method='bfill', limit_direction = "backward") # bffil
        return X # our transformed df


class OneHotEncoderWithNames(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.imputer = Custom_Cat_Imputer() # 
        self.one_hot_encoder = OneHotEncoder()
        self.column_names = None
        
    def get_rep_value(self, X): # deal with wrongly placed numeric objects in the object feature
        mode = X.mode().tolist()[0]        
        X = X.apply(lambda x : mode if OneHotEncoderWithNames.to_n(x) == "!" else x) # ! if it was numeric
        return X
        
    @staticmethod
    def to_n(x):
        try:
            int(x)
            return "!" # if we can concert to int
        except ValueError: # if no
            return x
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        global one_hot_names # write the names of the encoded features
        X_imputed = pd.DataFrame(self.imputer.fit_transform(X),columns = X.columns) # impute nulls
        X_imputed = X_imputed.apply(self.get_rep_value) # get rid of the "numeric" objects in feature
        X_one_hot_encoded = self.one_hot_encoder.fit_transform(X_imputed) # encode
        self.column_names = self.one_hot_encoder.get_feature_names(X_imputed.columns) # the encoded names
        X_df = pd.DataFrame(X_one_hot_encoded.toarray(), columns=self.column_names)
        one_hot_names = self.column_names
        return X_df


class null_imputer_corr(BaseEstimator, TransformerMixin): # based on the linear correlation between features
    def fit(self,X,y=None):
        self.corr_matrix = X.corr()
        return self
    def transform(self,X,y=None):
        X = pd.DataFrame(X).copy()
        for column in X.columns: # for each column
            col = list(self.corr_matrix[column]) # get the cor values
            vals = []
            for i,correlation in enumerate(col):
                vals.append((i,correlation)) 
            vals = sorted(vals,key = lambda y : y[1], reverse = True) # get the most correlated one
            for t in vals:
                if t[0] == i: # if it is us - ignore
                    continue
                else: # else, we have the max and break
                    val = t[0]
                    break
            med = X.iloc[:, val].median() # get the median of the correlated column
            X[column].fillna(med, inplace=True) # replace with the mdeian
        return X     
    

class null_imputer_ml(BaseEstimator, TransformerMixin): # ml approach
    def __init__(self,dec = False):
        self.data = {}
        self.dec = dec
    def fit(self,X,y=None):
        X = pd.DataFrame(X).copy().reset_index(drop=True)
        #features = set(X.columns.tolist()) 
        m = X.isna()
        length = m.shape[0]
        for col in X.columns:
            perc = sum(m[col])/length
            
            if perc <= 0.21:
                median = X[col].median()
                self.data[col] = median 
        return self
  
    def transform(self,X,y=None):
        X = pd.DataFrame(X).copy().reset_index(drop=True)    
        for col in list(self.data.keys()):
            median = self.data[col]
            x = X[col].copy()
            x = x.fillna(median)
            X[col] = x      
        features_clean = list(self.data.keys())
        features_null = list(set(X.columns).difference(set(features_clean)))      
        for col in features_null:
            ser = X[col]
            test = list(ser[ser.isnull()].index) 
            train = list(ser[ser.notnull()].index) 
            X_train = X[features_clean].iloc[train, :] 
            y_train = X[[col]].iloc[train, :]
            X_test = X[features_clean].iloc[test, :] 
            model = LinearRegression() if not self.dec else RandomForestRegressor() 
            model.fit(X_train,y_train) 
            preds = model.predict(X_test)
            no_nulls = []
            if not self.dec:
                for cube in preds: 
                    no_nulls.append(cube[0])
            else:
                no_nulls = preds            
            x = X.loc[:, col].copy() 
            x[test] = no_nulls 
            X.loc[:, col] = x   
        return X


        
class custom_numeric_imputer(BaseEstimator, TransformerMixin): # ml approach
    def __init__(self,dec = False):
        self.models = [] 
        self.tests = {}
        self.dec = dec
        
    def fit(self,X,y=None):
        features = set(X.columns.tolist()) 
        m = X.isna()
        select = []
        
        for col in list(features): # select only the cols with no nulls!
            if sum(m[col]) == 0:
                select.append(col)
                
        features = set(select) # these are our features 
        to_change = list(set(X.columns.tolist()).difference(features)) # columnns that do have at least one null
        
        for col in to_change: # for each null column
            ser = X[col]
            ser = ser.reset_index(drop=True) # we have to reset index...
            test = list(ser[ser.isnull()].index) # seperate nulls in our target
            train = list(ser[ser.notnull()].index) # seperate not nulls in our target
            """
            for our features get the non null target rows
            """
            X_train = X[list(features)].iloc[train, :] 
            y_train = X[[col]].iloc[train, :]
            model = LinearRegression() if not self.dec else RandomForestRegressor() #LinearRegression() # regression or dec tree
            model.fit(X_train,y_train) # train
            self.tests[col] = [model,test] # model, and test indicies
        else:
            self.features = list(features) # remembrt the list of features
        return self
            
    def transform(self,X,y=None):
        X = pd.DataFrame(X).copy().reset_index(drop=True)
        for col in X.columns: # for each column
            data = self.tests.get(col,False)
            if not data: # if False - continue
                continue
            model,test_inds = data # unpack the model and the null indicies
            no_nulls = []
            """
            Select the null rows in the target to predict
            """
            preds = model.predict(X[self.features].iloc[test_inds, :])
            if not self.dec:
                for cube in preds: # convert the result to one dim array
                    no_nulls.append(cube[0])
            else:
                no_nulls = preds
                
            x = X.loc[:, col].copy()  # get the copy
            x[test_inds] = no_nulls # replace the null indicies with the predicted not nulls
            X.loc[:, col] = x # make the change
        return X    
    

# the dates in the proper dates format (not numeric)
class Dates_common_Pipeline(BaseEstimator,TransformerMixin): # convert the thing to the object format!
    def fit(self,X,y=None): 
        return self
    def transform(self,X,y=None):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            X[col] = X[col].interpolate(method='pad', limit_direction = "forward")#X[col].interpolate(method='linear')
            X[col] = X[col].interpolate(method='bfill', limit_direction = "backward")    
        X[X.columns.tolist()] = (X[X.columns.tolist()] - pd.Timestamp('1970-01-01')) // pd.Timedelta('1s')
        return X 
    

  # the numeric dates
class Dates_numeric_Pipeline(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None): 
        return self
    def transform(self,X,y=None):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            X[col] = X[col].interpolate(method='linear').round(0)
            X[col] = X[col].interpolate(method='bfill', limit_direction = "backward")
        return X 
    
    
def deal_with_sqft(x):
    if x == None:
        return None
    if isinstance(x,int) or isinstance(x,float):
        return x
    try:
        if re.findall(r"\s*\d+\s*-\s*\d+\s*",x):
            a,b = x.split("-")
            a = int(a.strip())
            b = int(b.strip()) 
            a = min([a,b])
            b = max([a,b])
            mean = (a+b)/2
            low = (a-b)/2 
            up = (b-a)/2
            return mean+ random.uniform(low, up)
        elif re.findall(r"\s*\d+\s*\+\s*",x):
            num = float((x.strip()[:-1].strip()))
            return num + random.gauss(num, 20)
        elif re.findall(r"\s*[<]\s*\d+\s*",x):
            num = float(x.strip()[1:].strip())
            mean = num/2
            return mean+ random.uniform(0, mean-1)
        else:
            return float(x)
    except TypeError:
        return None