from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

X = [[1,2],[2,4],[4,5],[3,2],[3,1]] # Đầu vào
Y = [0,0,1,1,2] # Đầu ra

model = OneVsRestClassifier(estimator=SVC(random_state=0))
model.fit(X, y)
# Kết quả đoán nhận của điểm [3,5]
model.predict([3,5])