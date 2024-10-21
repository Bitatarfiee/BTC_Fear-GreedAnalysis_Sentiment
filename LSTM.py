from tensorflow.keras.layers import Input, Embedding, BatchNormalization, Dropout, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def lstm_model(Xtrain, Xval, ytrain, yval, V, D, maxlen, epochs):
    print("----Building the model----")
    i = Input(shape=(maxlen,))
    x = Embedding(V + 1, D)(i)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Conv1D(32, 5, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = MaxPooling1D(2)(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = LSTM(64)(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(i, x)
    model.summary()


    ytrain = np.array(ytrain).reshape(-1, 1)
    yval = np.array(yval).reshape(-1, 1)
    

    classes, counts = np.unique(ytrain, return_counts=True)
    print(f"Classes: {classes}")
    print(f"Counts: {counts}")
    

    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=ytrain.ravel())
    class_weights = dict(zip(classes, class_weights))
    print(f"Class weights: {class_weights}")


    print("----Training the network----")
    model.compile(optimizer=Adam(0.000007),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    r = model.fit(Xtrain, ytrain, 
                  validation_data=(Xval, yval), 
                  epochs=epochs, 
                  verbose=2,
                  batch_size=32,
                  class_weight=class_weights)
            
                  

    train_score = model.evaluate(Xtrain, ytrain, verbose=0)
    val_score = model.evaluate(Xval, yval, verbose=0)
    
    print(f"Train score: {train_score}")
    print(f"Validation score: {val_score}")
    
    n_epochs = len(r.history['loss'])
    
    return r, model, n_epochs