import numpy as np
import pandas as pd
import math
import scipy as sp
import matplotlib.pyplot as plt
import scipy
import pydotplus

from sklearn import datasets

## pre model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV

## preprocessing
from sklearn.preprocessing import LabelBinarizer

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.neighbors import KNeighborsRegressor

# Tree regressor and classifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# RF & GBoost
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier

###############################################################################
###############################################################################
###############################################################################

# The dataset also happens to be in the sklearn library
data = datasets.load_breast_cancer()

cd = data.DataFrame(data.data, columns= data.feature_names)



# Alt simple logit
logreg = sm.Logit(y_train, sm.add_constant(x_train))
#Instantiate model and print descriptive stats
logreg
result = logreg.fit()
stats1 = result.summary()
stats2 = result.summary2()
print(stats1)
print(stats2)


# ALT LOGIT

### logit
XX = sm.add_constant(x_train)
lfitM = sm.Logit(y_train, XX).fit() ##THIS GIVES SINGULAR ERROR LIKELY BECAUSE OF 0's
XXp = sm.add_constant(x_test)
phlgt = lfitM.predict(XXp)


##auc
auclgt = roc_curve(y_test,phlgt)
print('auc for logit: ',auclgt)

##roc
roclgt = roc_curve(y_test,phlgt)
plt.plot(roclgt[0],roclgt[1])
plt.xlabel('false positive rate'); plt.ylabel('true postive rate')
plt.title('logit auc ' + str(np.round(auclgt,2)))
plt.show()


#RF

# predict on test
#yhat_rf = rfc.predict(x_test)[:,1]   #### ISSUES

#plt.scatter(yhat_rf,x_test,c='green',s=.5)
#plt.xlabel('yhat on validation from rf on train');plt.ylabel('validation y')
#plt.show()




# Logit Binary Classification


#log reg 1, no L2 penalty fit on training data
logreg1 = LogisticRegression(penalty='none',random_state=0,multi_class='multinomial').fit(x_train_norm,y_train)

#print parameters to see tunable parameters
params = logreg1.get_params()
print(params)

print('Intercept: \n', logreg1.intercept_)
print('Coefficients: \n', np.exp(logreg1.coef_))


scores = cross_val_score(logreg1, x_train_norm, y_train, scoring='accuracy', cv=100).mean()

print("Average accuracy of 100 fold cross validation is %s" % round(scores*100,2))


##auc
phlgt = logreg1.predict(x_test_norm)
auclgt = roc_auc_score(y_test,phlgt)
print('auc for logit: ',auclgt)

roclgt = roc_curve(y_test,phlgt)
plt.plot(roclgt[0],roclgt[1])
plt.xlabel('false positive rate'); plt.ylabel('true postive rate')
plt.title('logit auc ' + str(np.round(auclgt,2)))
plt.show()

#log reg 2 with updated params



# Random Forest Classification

rfc = RandomForestClassifier(random_state=0,n_jobs=-1,n_estimators=50,max_features=2,oob_score=True)
rfc.fit(x_train_norm,y_train)
phrf = rfc.predict_proba(x_test_norm)[:,1]

# the OOB score is computed as the number of correctly predicted rows from the out of bag sample.
# not to useful in this application
print("oob score for Random Forest Classification: ",rfc.oob_score_)

##auc
aucrf = roc_auc_score(y_test,phrf)
print('auc for random forests: ',aucrf)

##roc
rocrf = roc_curve(y_test,phrf)
plt.plot(rocrf[0],rocrf[1])
plt.xlabel('false positive rate'); plt.ylabel('true postive rate')
plt.title('random forests auc ' + str(np.round(aucrf,2)))
plt.show()

scores = cross_val_score(rfc, x_train_norm, y_train, scoring='accuracy', cv=100).mean()

print("Average accuracy of 100 fold cross validation is %s" % round(scores*100,2))



# Trees

dtm = DecisionTreeClassifier(max_leaf_nodes=50)
dtm.fit(x_train_norm,y_train)
phdt = dtm.predict_proba(x_test_norm)[:,1]


##auc
aucrf = roc_auc_score(y_test,phdt)
print('auc for trees: ',aucrf)

##roc
rocrf = roc_curve(y_test,phrf)
plt.plot(rocrf[0],rocrf[1])
plt.xlabel('false positive rate'); plt.ylabel('true postive rate')
plt.title('trees auc ' + str(np.round(aucrf,2)))
plt.show()


scores = cross_val_score(dtm, x_train_norm, y_train, scoring='accuracy', cv=100).mean()

print("Average accuracy of 100 fold cross validation is %s" % round(scores*100,2))
