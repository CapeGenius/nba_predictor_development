import pandas
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn import svm
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report
from sklearn import model_selection
from imblearn.over_sampling import SMOTE, SVMSMOTE
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_blobs
from sklearn.inspection import DecisionBoundaryDisplay
import seaborn as sns


def pca(data_array):
    # calculate the covariance matrix
    covariance_matrix = np.cov(np.transpose(data_array))
    w, eigenvectors = np.linalg.eig(covariance_matrix)
    eigenvectors = np.real_if_close(eigenvectors, tol=1)

    # tranposes eigenvectors
    new_eigenvectors = preprocessing.normalize(np.transpose(eigenvectors))

    # gets each principal component
    for i in range(k):
        principal_components[i] = new_eigenvectors[i]

    # projected NBA team data onto these principal components
    projected_components = np.transpose(
        (data_array @ np.transpose(principal_components))
    )

    return projected_components.tolist()


# creates the test/train data by adding a champion or nonchampion result
def create_train_test(index_dict, projected_components_list):

    result_list = []
    print("The size is", len(projected_components_list[0]))

    # adds another column that identifies 0 with nonchampioon team, and 1 with champion team
    for x in range(725):
        # print(x)
        if x in index_dict:
            result_list.append(1)
        else:
            result_list.append(0)

    print("Length of 1", len(result_list))
    print("Length of 2", len(projected_components_list[0]))

    # add results to principal components lists and adds test train results (0 or 1)
    projected_components_list.append(result_list)
    test_train_array = np.array(projected_components_list)
    test_train_array = np.transpose(test_train_array)

    return test_train_array


# declaring data structures and variables
all_team_lists = [[]]
index_lists = []
k = 4
principal_components = [[0] * 725] * k

# reading csv data in dataframe and then data structure
python_team_data = pandas.read_csv("PythonRawData.csv")
all_team_data = pandas.read_csv("AllTeamData.csv")
index_data = pandas.read_csv("Index.csv")
season_2024_data = pandas.read_csv("season20232024.csv")

print(season_2024_data)

# turning data frames into list
all_team_lists = all_team_data.values.tolist()
python_team_lists = python_team_data.values.tolist()
season_2024_lists = season_2024_data.values.tolist()

index_lists = np.transpose(np.array(index_data.values.tolist()))
index_lists = index_lists.tolist()
index_lists = [item for sublist in index_lists for item in sublist]

# print(index_lists)

# convert dictionary
champion_dict = dict.fromkeys(index_lists, None)
# print(champion_dict)

# delete the first column
for row in all_team_lists:
    del row[0]

# all_teams_lists = all_team_lists[:][:]

# print(all_team_lists[:][:])
all_team_array = np.array(all_team_lists)
python_team_array = np.array(python_team_lists)
season_2024_array = np.array(season_2024_lists)

projected_components = pca(all_team_array)
python_team_components = pca(python_team_array)
projected_2024_components = pca(season_2024_array)

# creating two separate data structures for projected components
final_component_1 = projected_components[0]
final_component_2 = projected_components[1]

# separating championship components from nonchampion components
champion_components_1 = final_component_1[0:24]
champion_components_2 = final_component_2[0:24]

nonchampion_components_1 = final_component_1[25:724]
nonchampion_components_2 = final_component_2[25:724]

# plotting the champion vs non champion PCA graph
plt.scatter(champion_components_1, champion_components_2, s=10)
plt.scatter(nonchampion_components_1, nonchampion_components_2, s=5)
plt.scatter(projected_2024_components[1], projected_2024_components[2], s=5)
plt.title("Primary Component vs Secondary Component")
plt.xlabel("Primary Component")
plt.ylabel("Secondary Component")
plt.legend(["Champion Teams", "Nonchampion Teams", "2023 Teams"])
plt.show()

# print(python_team_components)

# creates a train/test dataset
test_train_array = create_train_test(champion_dict, python_team_components)

# best_h_fit = optimized_SVC(test_train_array)

columns2 = list()

# creates column name for k eigenvectors
for i in range(k):
    columns2.append("Eigenvector " + str(i + 1))
print(columns2)
columns2.append("Result")
print(columns2)


# creates a dataframe and identifies independent/dependent variables
test_train_dataframe = pandas.DataFrame(test_train_array, columns=columns2)
independent_variables = test_train_dataframe[columns2[0:k]]
dependent_variables = test_train_dataframe["Result"]
projected_2024_dataframe = pandas.DataFrame(projected_2024_components)


# creates test and train set
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    independent_variables, dependent_variables, test_size=0.33, random_state=100
)

# resampling the data
sm = SMOTE(random_state=10, sampling_strategy=0.8)
sm2 = SVMSMOTE(random_state=10, sampling_strategy=0.6, k_neighbors=5)
print("sm is ", sm)

# oversample the data
x_res, y_res = sm2.fit_resample(X_train, y_train)

# parameter grid for hyperparameter testing
param_grid = {"C": range(1, 100), "random_state": range(1, 100)}

# apply a linear svc to the classification
h1 = svm.LinearSVC(C=26, class_weight={0: 5, 1: 200})


# grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
# h1 = GridSearchCV(svm.LinearSVC(), param_grid, refit = True, verbose = 3, scoring="precision")

# x_res, y_res prediction
h1.fit(x_res, y_res)
h1.score(x_res, y_res)

# predict data and cross tab
y_pred = h1.predict(X_test)
season_2024_pred = h1.predict(
    np.transpose(np.array(projected_2024_dataframe.values.tolist()))
)
pandas.crosstab(y_test, y_pred)

# show a confusion matrix
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix, display_labels=["Nonchampions", "Champions"]
)
cm_display.plot()
plt.title("Confusion Matrix for SVM")
plt.show()

# shows the classification results by cross tabing the prediction vs the test
print("Y_pred is ", y_pred)
print("The cross tab is ", pandas.crosstab(y_test, y_pred))
print(classification_report(y_test, y_pred))
# print("The next classification report is ", classification_report(y_test, y_pred_2) )
# print("The best estimator is ",h1.best_estimator_)
# print("The best params is ",h1.best_params_)

X_1, y_1 = make_blobs(n_samples=40, centers=2, random_state=0)
C = 26

print("our prediction is", season_2024_pred)
