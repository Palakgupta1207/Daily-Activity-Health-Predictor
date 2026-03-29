import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

# 1. Load Dataset

data = pd.read_csv("data/health.csv")

print("\nDataset Loaded:\n")
print(data.head())


# 2. Features & Target

X = data.drop("health", axis=1)
y = data["health"]


# 3. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 4. Train Model

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# 5. Evaluate

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")


# 6. Feature Importance

features = X.columns
importance = model.coef_[0]

print("\nFeature Importance:")
for f, imp in zip(features, importance):
    print(f"{f}: {round(imp, 3)}")


# 7. Suggestions Function

def give_suggestions(steps, sleep, screen, water, exercise, junk):
    tips = []

    if steps < 7000:
        tips.append(" Increase daily steps.")
    if sleep < 7:
        tips.append(" Improve sleep (7–8 hrs recommended).")
    if screen > 5:
        tips.append(" Reduce screen time.")
    if water < 2:
        tips.append(" Drink more water.")
    if exercise == 0:
        tips.append(" Add some exercise.")
    if junk == 1:
        tips.append(" Reduce junk food.")

    return tips


# 8. Save History

def save_history(data_row):
    file_path = "history.csv"

    df = pd.DataFrame([data_row])

    if not os.path.exists(file_path):
        df.to_csv(file_path, index=False)
    else:
        df.to_csv(file_path, mode='a', header=False, index=False)


# 9. Prediction Function

def predict_health():
    try:
        print("\nEnter your daily data:")

        steps = int(input("Steps walked: "))
        sleep = float(input("Sleep hours: "))
        screen = float(input("Screen time (hrs): "))
        water = float(input("Water intake (liters): "))
        exercise = int(input("Exercise (1=yes, 0=no): "))
        junk = int(input("Junk food (1=yes, 0=no): "))

        input_data = [[steps, sleep, screen, water, exercise, junk]]

        # Prediction
        result = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        # Health Score
        score = round(prob * 100, 2)

        print(f"\n Health Score: {score}/100")

        if result == 1:
            print(" Healthy Lifestyle")
        else:
            print(" Unhealthy Lifestyle")

        # Suggestions
        tips = give_suggestions(steps, sleep, screen, water, exercise, junk)

        if tips:
            print("\n Suggestions:")
            for t in tips:
                print("-", t)
        else:
            print("\n Great job! Keep it up!")

        # Save history
        save_history({
            "steps": steps,
            "sleep": sleep,
            "screen": screen,
            "water": water,
            "exercise": exercise,
            "junk": junk,
            "score": score
        })

    except:
        print("\n Invalid input! Try again.")


# 10. Run Loop

while True:
    predict_health()

    again = input("\nCheck again? (yes/no): ")
    if again.lower() != "yes":
        print("Program Ended.")
        break
