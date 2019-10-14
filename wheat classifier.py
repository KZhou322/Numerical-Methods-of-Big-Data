import numpy as np
import plotly.graph_objects as pgo

matrix = np.ones([1,8])
with open("seeds.tsv") as f:
    for line in f:
        fields = line.split()
        floatFields=[float(i) for i in fields]
        arrayconvert = np.asarray(floatFields)
        matrix = np.vstack((matrix,arrayconvert))

#establishing the feature and ground truth matrices
matrix = np.delete(matrix,0,0)
matrix = np.split(matrix, [7],1)
features=matrix[0].astype(np.float).T
gtruth = matrix[1].astype(np.int).T
gtruth = gtruth[0]

#normalizing the data with 0 mean and unit variance
features_meanzero = features - np.mean(features, axis=1, keepdims=True)
features_standardized = features_meanzero/np.std(features_meanzero, axis = 1, keepdims=True)
featuresM2 = features_standardized + np.random.normal(size=features_standardized.shape, scale = 0.2)
featuresM3 = features_standardized + np.random.normal(size=features_standardized.shape, scale = 1.5)

#computing all the svds for all the matrices
U, S, VT = np.linalg.svd(features_standardized)
U2, S2, VT2 = np.linalg.svd(featuresM2)
U3, S3, VT3 = np.linalg.svd(featuresM3)

## --showing the variances of the principal components
# variances = S**2/(210-1)
#
# svd_figure = pgo.Figure([pgo.Bar(x=np.arange(len(S)), y=variances)])
# svd_figure.show()

## --projecting the dataset onto the first two principal components
features_pcbasis = U.T @ features_standardized
features_pcbasis2 = U.T @ featuresM2
features_pcbasis3 = U.T @ featuresM3
#
# projection = pgo.Figure(data=pgo.Scatter(x=features_pcbasis[0], y=features_pcbasis[1], mode="markers", marker=dict(
#         size=16,
#         color=gtruth,
#         colorscale = "Viridis",
#         showscale=True
#     )))
# projection.update_layout(showlegend=False)
# projection.show()

## --showing the measurements that contribute most to the first two principal components
#print(U[:,0], U[:,1])

## --Error checking the proposed rule
# error = 0
# for check in range(len(gtruth)):
#     if gtruth[check] == 1 and (features_pcbasis[0][check]<-1 or features_pcbasis[0][check]>1) :
#         error+=1
#     elif gtruth[check]==2 and (features_pcbasis[0][check]>-1):
#         error+=1
#     elif gtruth[check]==3 and (features_pcbasis[0][check]<1):
#         error+=1
# print (error)

## --Error checking the 0.2 perturbed with the proposed rule
# error = 0
# for check in range(len(gtruth)):
#     if gtruth[check] == 1 and (features_pcbasis2[0][check]<-1 or features_pcbasis2[0][check]>1) :
#         error+=1
#     elif gtruth[check]==2 and (features_pcbasis2[0][check]>-1):
#         error+=1
#     elif gtruth[check]==3 and (features_pcbasis2[0][check]<1):
#         error+=1
# print (error)

## --Error checking the 1.5 perturbed with the proposed rule
# error = 0
# for check in range(len(gtruth)):
#     if gtruth[check] == 1 and (features_pcbasis3[0][check]<-1 or features_pcbasis3[0][check]>1) :
#         error+=1
#     elif gtruth[check]==2 and (features_pcbasis3[0][check]>-1):
#         error+=1
#     elif gtruth[check]==3 and (features_pcbasis3[0][check]<1):
#         error+=1
# print (error)