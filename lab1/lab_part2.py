import numpy as np

# Вхідні дані
data = [
    1.19, 5.61, 0.89, 6.00, 1.04, 5.98, 0.03, 6.00, 1.83, 4.23, 0.60, 4.15, 0.13, 5.01, 1.87
]

# Розділення даних на тренувальний та тестовий набори
train_data = data[:-2]  # Перші 13 чисел як тренувальний набір
test_data = data[-2:]   # Останні два числа як тестовий набір

print("Train data:", train_data)
print("Test data:", test_data)

# Очікуваний результат
expected_output = [6.00, 1.04, 5.98, 0.03, 6.00, 1.83, 4.23, 0.60, 4.15, 0.13, 5.01, 1.87]

class SingleNeuronNetwork:
    def __init__(self):
        self.weights = np.ones(3)  # Всі вагові коефіцієнти рівні 1

    def weighted_sum(self, inputs):
        return np.dot(inputs, self.weights)

    def activation_function(self, weighted_sum):
        return 1 / (1 + np.exp(-weighted_sum)) * 10

    def train(self, inputs, expected_output, learning_rate=0.01, epochs=5000, tolerance=0.0001):
        
        for epoch in range(epochs):
            total_error = 0
            previous_error = float('inf')

            for i in range(len(inputs) - 2):
                input_sequence = inputs[i:i + 3]
                expected_output_value = expected_output[i]

                weighted_sum = self.weighted_sum(input_sequence)
                output = self.activation_function(weighted_sum)

                error = expected_output_value - output
                total_error += error ** 2

                derivative = error * (np.exp(-weighted_sum) / (1 + np.exp(-weighted_sum)) ** 2)
                corrections = learning_rate * derivative

                self.weights += np.array(input_sequence) * corrections

            mean_squared_error = total_error / len(inputs)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Mean squared error: {mean_squared_error}")

            if abs(previous_error - mean_squared_error) < tolerance:
                break
            previous_error = mean_squared_error

        return self.weights

    def test(self, test_inputs):
        predictions = []
        
        for i in range(len(test_inputs) - 2):
            input_sequence = test_inputs[i:i + 3]
            weighted_sum = self.weighted_sum(input_sequence)
            output = self.activation_function(weighted_sum)
            predictions.append(output)
        
        return predictions

# Створення екземпляру класу
neuron_network = SingleNeuronNetwork()

# Навчання мережі на тренувальних даних
neuron_network.train(train_data, expected_output)

# Тестування мережі на тренувальних та тестових даних
train_predictions = neuron_network.test(train_data)
test_predictions = neuron_network.test(train_data[-5:])

# Обчислення середньоквадратичної помилки
def calculate_mse(predictions, true_values):
    squared_errors = [(p - t) ** 2 for p, t in zip(predictions, true_values)]
    mse = sum(squared_errors) / len(true_values)
    return mse

def evaluate_predictions(predictions, true_values):
    print("True values | Predicted values | Precision")
    print("--------------------------------------------")
    precisions = [p - t for p, t in zip(predictions, true_values)]
    for true_val, pred, prec in zip(true_values, predictions, precisions):
        print(f"{true_val:11.2f} | {pred:14.10f} | {prec:.2f}")
    mse = calculate_mse(predictions, true_values)
    print("--------------------------------------------")
    print("Mean Squared Error:", mse)

# Використання функції для оцінки прогнозів на тренувальних даних
print("Evaluation on training data:")
evaluate_predictions(train_predictions, expected_output[:len(train_predictions)])

# Використання функції для оцінки прогнозів на тестових даних
print("Evaluation on test data:")
evaluate_predictions(test_predictions, test_data[:len(test_predictions)])
