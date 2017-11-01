#Azure
setwd("~/Analytics Vidya/Target the right customer")

#Office
setwd('C:\\Users\\bthanish\\Desktop\\Thanish\\Analytics Vidya\\Target the right customer')

library(data.table)

train_prod = fread('train.csv')
test_prod = fread('test_plBmD8c.csv')

#Merging the train and test
test_prod[, RESPONDERS:= NA]
train_test_prod = rbind(train_prod, test_prod)

#Drop off the constant columns
train_test_prod[, ':='(OCCUP_ALL_NEW = NULL, 
                       PM_FD_MON_04 = NULL, 
                       STMT_CON_DAE_CLOSED_MON_01 = NULL, 
                       EEG_TAG = NULL, 
                       PM_FD_MON_02 = NULL, 
                       MER_EMI_CLOSED_MON_01 = NULL, 
                       STMT_CON_DAE_ACTIVE_MON_01 = NULL, 
                       EEG_CLOSED = NULL
)]

#Extracting the day, month and year from 
#MATURITY_GL, MATURITY_LAP, MATURITY_LAS, CLOSED_DATE
train_test_prod[, ':=' (MATURITY_GL_day = as.numeric(substr(x = MATURITY_GL, start = 1, stop = 2)),
                        MATURITY_GL_month = substr(x = MATURITY_GL, start = 3, stop = 5),
                        MATURITY_GL_year = as.numeric(substr(x = MATURITY_GL, start = 6, stop = 9)),
                        MATURITY_LAP_day = as.numeric(substr(x = MATURITY_LAP, start = 1, stop = 2)),
                        MATURITY_LAP_month = substr(x = MATURITY_LAP, start = 3, stop = 5),
                        MATURITY_LAP_year = as.numeric(substr(x = MATURITY_LAP, start = 6, stop = 9)),
                        MATURITY_LAS_day = as.numeric(substr(x = MATURITY_LAS, start = 1, stop = 2)),
                        MATURITY_LAS_month = substr(x = MATURITY_LAS, start = 3, stop = 5),
                        MATURITY_LAS_year = as.numeric(substr(x = MATURITY_LAS, start = 6, stop = 9)),
                        CLOSED_DATE_day = as.numeric(substr(x = CLOSED_DATE, start = 1, stop = 2)),
                        CLOSED_DATE_month = substr(x = CLOSED_DATE, start = 3, stop = 5),
                        CLOSED_DATE_year = as.numeric(substr(x = CLOSED_DATE, start = 6, stop = 9)))]

#Filling up the NA and the empty columns 
train_test_prod[is.na(train_test_prod)] = -999
train_test_prod[train_test_prod==''] = 'Not sure'

#Fixing the zip column 
train_test_prod[, ZIP_CODE_FINAL := as.numeric(as.character(ZIP_CODE_FINAL))]
train_test_prod[is.na(ZIP_CODE_FINAL), ZIP_CODE_FINAL:=-999]

#Separating the integer and factor columns
fac_columns = colnames(train_test_prod)[sapply(X=train_test_prod, FUN = function(x)({class(x) =='character'}))]
int_columns = setdiff(colnames(train_test_prod), fac_columns)

#Convert the character columns to numerics
train_test_prod[, (fac_columns) := lapply(.SD, as.factor), .SDcols = fac_columns]

#Splitting it back to train and test
train_prod = train_test_prod[RESPONDERS != -999,]
train_prod[, RESPONDERS:= as.factor(as.character(RESPONDERS))]
test_prod = train_test_prod[RESPONDERS == -999,]
test_prod[, RESPONDERS:=NULL]
sub_test_id = test_prod$CUSTOMER_ID
test_prod[,CUSTOMER_ID := NULL]
train_prod[,CUSTOMER_ID := NULL]

#Split into local train and test
set.seed(100)
sample_split = sample(nrow(train_prod), 0.6*nrow(train_prod), replace=F)
train_local = train_prod[sample_split,]
test_local = train_prod[-sample_split,]

#XGB
library(xgboost)

train_local[, (fac_columns) := lapply(.SD, as.numeric), .SDcols = fac_columns]
test_local[, (fac_columns) := lapply(.SD, as.numeric), .SDcols = fac_columns]
train_prod[, (fac_columns) := lapply(.SD, as.numeric), .SDcols = fac_columns]
test_col = setdiff(fac_columns, 'RESPONDERS')
test_prod[, (test_col) := lapply(.SD, as.numeric), .SDcols = test_col]

train_prod[, RESPONDERS := RESPONDERS-1]
train_local[, RESPONDERS := RESPONDERS-1]
test_local[, RESPONDERS := RESPONDERS-1]

x_indep = setdiff(colnames(train_local), c('pre_appv_sal','RESPONDERS'))
y_dep = 'RESPONDERS'

dtrain_local = xgb.DMatrix(data = as.matrix(train_local[,x_indep, with=F]), label = as.matrix(train_local[,y_dep, with=F]))
dtest_local = xgb.DMatrix(data = as.matrix(test_local[,x_indep, with=F]), label = as.matrix(test_local[,y_dep, with=F]))
dtrain_prod = xgb.DMatrix(data = as.matrix(train_prod[,x_indep, with=F]), label = as.matrix(train_prod[,y_dep, with=F]))
dtest_prod = xgb.DMatrix(data = as.matrix(test_prod[,x_indep, with=F]))

watchlist = list(train = dtrain_local, test = dtest_local)

#XGB_1 - random
xgb.local.model = xgb.train(data = dtrain_prod, 
                            watchlist = watchlist, 
                            eta = 0.1,
                            nrounds = 110,
                            max_depth = 6,
                            subsample =0.9,
                            colsample_bytree =1,
                            objective = "binary:logistic",
                            eval_metric = 'auc',
                            early_stopping_rounds = 30,
                            maximize = T)
xgb.prod.pred.1 = predict(xgb.local.model, newdata = dtest_prod)


#XGB_2 - random
xgb.local.model = xgb.train(data = dtrain_prod, 
                            watchlist = watchlist, 
                            eta = 0.1,
                            nrounds = 110,
                            max_depth = 6,
                            subsample =0.9,
                            colsample_bytree =1,
                            objective = "binary:logistic",
                            eval_metric = 'auc',
                            early_stopping_rounds = 30,
                            maximize = T)
xgb.prod.pred.2 = predict(xgb.local.model, newdata = dtest_prod)


#XGB_3
set.seed(100)
xgb.local.model = xgb.train(data = dtrain_prod, 
                            watchlist = watchlist, 
                            eta = 0.1,
                            nrounds = 110,
                            max_depth = 6,
                            subsample =0.9,
                            colsample_bytree =1,
                            objective = "binary:logistic",
                            eval_metric = 'auc',
                            early_stopping_rounds = 30,
                            maximize = T)
xgb.prod.pred.3 = predict(xgb.local.model, newdata = dtest_prod)


#XGB_4
set.seed(75)
xgb.local.model = xgb.train(data = dtrain_prod, 
                            watchlist = watchlist, 
                            eta = 0.1,
                            nrounds = 110,
                            max_depth = 6,
                            subsample =0.9,
                            colsample_bytree =1,
                            objective = "binary:logistic",
                            eval_metric = 'auc',
                            early_stopping_rounds = 30,
                            maximize = T)
xgb.prod.pred.4 = predict(xgb.local.model, newdata = dtest_prod)


#XGB_5
set.seed(125)
xgb.local.model = xgb.train(data = dtrain_prod, 
                            watchlist = watchlist, 
                            eta = 0.1,
                            nrounds = 110,
                            max_depth = 6,
                            subsample =0.9,
                            colsample_bytree =1,
                            objective = "binary:logistic",
                            eval_metric = 'auc',
                            early_stopping_rounds = 30,
                            maximize = T)
xgb.prod.pred.5 = predict(xgb.local.model, newdata = dtest_prod)


#XGB_6
set.seed(150)
xgb.local.model = xgb.train(data = dtrain_prod, 
                            watchlist = watchlist, 
                            eta = 0.1,
                            nrounds = 110,
                            max_depth = 6,
                            subsample =0.9,
                            colsample_bytree =1,
                            objective = "binary:logistic",
                            eval_metric = 'auc',
                            early_stopping_rounds = 30,
                            maximize = T)
xgb.prod.pred.6 = predict(xgb.local.model, newdata = dtest_prod)

consol_df = data.frame(XGB_1 = xgb.prod.pred.1,
                       XGB_2 = xgb.prod.pred.2,
                       XGB_3 = xgb.prod.pred.3,
                       XGB_4 = xgb.prod.pred.4,
                       XGB_5 = xgb.prod.pred.5,
                       XGB_6 = xgb.prod.pred.6)

mean_pred = rowMeans(consol_df)

XGB_sub = data.frame(CUSTOMER_ID = sub_test_id,	RESPONDERS = mean_pred)
write.csv(XGB_sub, row.names=F, 'XGB_93_94_95_96_97_98_ens.csv')
length(xgb.prod.pred)

imp_df = as.data.frame(xgb.importance(model = xgb.local.model, feature_names = x_indep))

################################################################
