import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import keras_tuner as kt
import tensorflow_probability as tfp
import tensorflow as tf
from sklearn.utils import class_weight

# Load the data
df = pd.read_csv(
    "/home/swisnoski/nba_predictor_development/models/data/combined_data_2010-2023.csv"
)

# Replace win/loss indicators with binary values
df["TEAM_1_WIN/LOSS"] = df["TEAM_1_WIN/LOSS"].replace({100: 1, 0: 0})

# Drop rows where the target variable contains NaN
df = df.dropna(subset=["TEAM_1_WIN/LOSS"])

# Define features and target
X = df[
    [
        "TEAM_1_HOME/AWAY",
        "TEAM_1_DEF_PPP",
        "TEAM_1_TS%",
        "TEAM_1_eFG%",
        "TEAM_1_FG_PCT",
        "TEAM_1_DREB",
        "TEAM_1_AST",
        "TEAM_1_TOV",
        "TEAM_1_WIN_PCT",
        "TEAM_2_HOME/AWAY",
        "TEAM_2_DEF_PPP",
        "TEAM_2_TS%",
        "TEAM_2_eFG%",
        "TEAM_2_FG_PCT",
        "TEAM_2_DREB",
        "TEAM_2_AST",
        "TEAM_2_TOV",
        "TEAM_2_WIN_PCT",
    ]
]
y = df["TEAM_1_WIN/LOSS"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# One-hot encode the target variable
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


# Define the hypermodel with Mixture Density Network
def build_model(hp):
    model = Sequential()
    model.add(
        Dense(
            units=hp.Int("units", min_value=16, max_value=128, step=16),
            activation=hp.Choice("activation", values=["relu", "tanh"]),
            input_shape=(X_train.shape[1],),
        )
    )
    model.add(Dropout(hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.1)))
    model.add(
        Dense(
            units=hp.Int("units", min_value=16, max_value=128, step=16),
            activation=hp.Choice("activation", values=["relu", "tanh"]),
        )
    )
    model.add(Dropout(hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.1)))

    # Define the parameters for the mixture density network
    num_components = hp.Int("num_components", min_value=2, max_value=10, step=1)
    model.add(
        tf.keras.layers.Dense(num_components * 3)
    )  # 3 parameters per component: mean, stddev, and weight

    return model


# Custom loss function for Mixture Density Network
def mdn_loss(num_components):
    def loss(y_true, y_pred):
        y_true = tf.expand_dims(y_true, axis=-1)
        gm = tfp.distributions.MixtureSameFamily(
            mixture_distribution=tfp.distributions.Categorical(
                logits=y_pred[:, :num_components]
            ),
            components_distribution=tfp.distributions.Normal(
                loc=y_pred[:, num_components : num_components * 2],
                scale=tf.nn.softplus(y_pred[:, num_components * 2 :]),
            ),
        )
        return -gm.log_prob(y_true)

    return loss


# Initialize the tuner
tuner = kt.RandomSearch(
    build_model,
    objective="val_loss",
    max_trials=3,
    executions_per_trial=2,
    directory="my_dir",
    project_name="nba_mdn_tuning",
)

# Search for the best hyperparameters
tuner.search(X_train_resampled, y_train_resampled, epochs=50, validation_split=0.2)

# Get the best model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)

# Compile the model with the custom loss
num_components = best_hps.get("num_components")
model.compile(optimizer="adam", loss=mdn_loss(num_components))  # Default optimizer

# Train the best model
history = model.fit(
    X_train_resampled, y_train_resampled, epochs=50, validation_split=0.2, verbose=1
)

# Evaluate the model on the test set
loss = model.evaluate(X_test, y_test, verbose=0)
print(f"Best Mixture Density Network loss: {loss}")

# Make predictions
y_pred = model.predict(X_test)

# Convert predictions to categorical outputs for evaluation
gm = tfp.distributions.MixtureSameFamily(
    mixture_distribution=tfp.distributions.Categorical(
        logits=y_pred[:, :num_components]
    ),
    components_distribution=tfp.distributions.Normal(
        loc=y_pred[:, num_components : num_components * 2],
        scale=tf.nn.softplus(y_pred[:, num_components * 2 :]),
    ),
)
y_pred_proba = gm.mean().numpy()  # Convert Tensor to NumPy array

# Ensure exactly half wins and half losses
num_samples = len(y_pred_proba)
num_wins = num_samples // 2
num_losses = num_samples - num_wins

# Get indices for the top probabilities for wins
indices_sorted = np.argsort(y_pred_proba.flatten())
indices_wins = indices_sorted[-num_wins:]
indices_losses = indices_sorted[:-num_wins]

# Create the final predictions
y_pred_final = np.zeros(num_samples)
y_pred_final[indices_wins] = 1  # Mark top probabilities as wins
# Remaining are losses

# Convert y_test from one-hot encoded format to single integer labels
y_test_single = np.argmax(y_test, axis=1)

# Evaluate the model
print("Best Mixture Density Network classification report:")
print(classification_report(y_test_single, y_pred_final))

# Confusion matrix
conf_matrix_mdn = confusion_matrix(y_test_single, y_pred_final)
sns.heatmap(conf_matrix_mdn, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
