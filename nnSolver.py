import pandas as pd
import tensorflow as tf
import numpy as np

#https://medium.com/analytics-vidhya/how-to-solve-sudoku-with-convolutional-neural-networks-cnn-c92be8830c52
# https://www.tensorflow.org/tutorials/images/cnn

if __name__ == '__main__':
    # read the first 100 lines from sudoku.csv file into a pandas dataframe
    data = pd.read_csv('sudoku.csv', header=None, nrows=1000)
    # rename the columns of the dataframe
    data.columns = ['Puzzle', 'Solution']
    # delete first row of the dataframe
    data = data.drop(0)
    # print header of the dataframe
    print(data.head())

    puzzles = np.array([list(map(int, p)) for p in data['Puzzle']])
    solutions = np.array([list(map(int, s)) for s in data['Solution']])


    puzzles = puzzles.reshape((-1, 81, 1,1))
    # solutions = solutions.reshape((-1, 81, 1))
    solutions_81x9 = np.zeros((len(solutions), 81, 9))
    for i in range(len(solutions)):
        for j in range(81):
            if solutions[i, j] != 0:
                solutions_81x9[i, j, solutions[i, j]-1] = 1
    solutions = solutions.reshape((-1, 81))

    # mapped_solutions = []
    # for row in solutions:
    #     board_81x9 = np.zeros((81, 9))

    #     for i in range(81):
    #         board_81x9[i, row[i] - 1] = 1
    #     mapped_solutions.append(board_81x9)

    split_index = int(len(puzzles) * 0.8)
    train_puzzles = puzzles[:split_index]
    train_solutions = solutions_81x9[:split_index]

    val_puzzles = puzzles[split_index:]
    val_solutions = solutions_81x9[split_index:]

    #print(train_solutions[0])
    #print(val_solutions[0])

    # Define the model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', input_shape=(81, 1, 1)),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        #tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),

        # !!! solution that worked
        # tf.keras.layers.Conv2D(1, (1, 1), activation='relu', padding='same', input_shape=(81,)),

        # tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(3, 3), activation='relu', padding='same'),
        # tf.keras.layers.Conv2DTranspose(1, (3, 3), strides=(3, 3), activation='relu', padding='same'),
        # tf.keras.layers.Flatten(),
        # tf.keras.layers.Dense(81, activation='softmax')

        # tf.keras.layers.Conv1DTranspose(64, 3, strides=3, activation='relu', padding='same'),
        # tf.keras.layers.Conv1DTranspose(9, 3, strides=3, activation='linear', padding='same'),
        tf.keras.layers.Conv2D(9, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Reshape((81, 9)),
        tf.keras.layers.Softmax()
        
    ])

    model.summary()

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(train_puzzles, train_solutions, validation_data=(val_puzzles, val_solutions), epochs=10, batch_size=32)


    # Test the model
    test_puzzle = puzzles[0]
    test_solution = solutions[0]
    predicted_solution = model.predict(test_puzzle[np.newaxis, ...])[0]
    print('Test puzzle:')
    print(test_puzzle.reshape((9, 9)))
    print('Test solution:')
    print(test_solution.reshape((9, 9)))
    print('Predicted solution:')
    prediction = np.ones((81, 1))
    for i in range(81):
        prediction[i] = np.argmax(predicted_solution[i]) + 1
    print(prediction.reshape((9, 9)))



    
    
# # Split the data into training and validation sets
# split_index = int(len(puzzles) * 0.8)
# train_puzzles = puzzles[:split_index]
# train_solutions = solutions[:split_index]
# val_puzzles = puzzles[split_index:]
# val_solutions = solutions[split_index:]

# # Define the model architecture
# model = tf.keras.Sequential([
#     tf.keras.layers.Reshape((9, 9, 1), input_shape=(81,)),
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
#     tf.keras.layers.Conv2D(128, (1, 1), activation='relu', padding='same'),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(81, activation='softmax')
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# # Train the model
# for i in range(81):
#     # Select the ith blank in the puzzle
#     train_puzzles_i = train_puzzles[:, :]
#     train_puzzles_i[:, i] = 0

#     train_solutions_i = train_solutions[:, i]

#     val_puzzles_i = val_puzzles[:, :]
#     val_puzzles_i[:, i] = 0

#     val_solutions_i = val_solutions[:, i]

#     # Train the model on the ith blank
#     model.fit(train_puzzles_i, train_solutions_i, validation_data=(val_puzzles_i, val_solutions_i), epochs=1, batch_size=32)

#     # Test the model
#     test_puzzle = '530070000600195000098000060800060003400803001700020006060000280000419005000080079'
#     test_puzzle = np.array(list(map(int, test_puzzle))).reshape((9, 9))
#     predicted_solution = np.zeros((9, 9))
#     for i in range(81):
#         if test_puzzle.flat[i] == 0:
#             # Select the ith blank in the puzzle
#             test_puzzle_i = test_puzzle.copy()
#             test_puzzle_i.flat[i] = 0

#             # Predict the value to fill in the ith blank
#             predicted_value = np.argmax(model.predict(test_puzzle_i[np.newaxis, ...]), axis=1)

#             # Fill in the value in the predicted solution
#             predicted_solution.flat[i] = predicted_value

#     print(predicted_solution)













