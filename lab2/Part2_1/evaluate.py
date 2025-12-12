#!/usr/bin/env python3

# Copyright 2021 (c) Pierre-Emmanuel Novac <penovac@unice.fr> Université Côte d'Azur, CNRS, LEAT. All rights reserved.
# Updated for Neural Network Training

from collections import namedtuple
import sys
import time
import statistics
import serial
import matplotlib.pyplot as plt
import numpy as np
import os

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
except ImportError:
    print("Error: TensorFlow is required for training. Install it via: pip install tensorflow")
    sys.exit(1)

def load_dataset(x_path, y_path):
    """
    Robust data loader that handles both CSV (comma) and UCI HAR default (whitespace)
    """
    print(f"Loading data from {x_path} and {y_path}...")
    x_data = []
    y_data = []

    # Load Features (X)
    with open(x_path, 'r') as f:
        for line in f:
            # Handle both comma (CSV) and whitespace (TXT) delimiters
            if ',' in line:
                values = [float(x) for x in line.strip().split(',')]
            else:
                values = [float(x) for x in line.strip().split()]
            x_data.append(values)

    # Load Labels (Y)
    with open(y_path, 'r') as f:
        for line in f:
            line = line.strip()
            if ',' in line:
                # Assuming One-Hot CSV
                vals = [float(v) for v in line.split(',')]
                y_data.append(np.argmax(vals))
            else:
                # Assuming standard UCI HAR integer labels (1-6)
                # We subtract 1 to make classes 0-5 for the Neural Network
                y_data.append(int(line) - 1)

    return np.array(x_data), np.array(y_data)

def train(x_path, y_path, epochs: int=20, batch_size: int=32, learning_rate: float=0.001):
    """
    Train a Neural Network (MLP) using TensorFlow/Keras
    """
    epochs = int(epochs)
    batch_size = int(batch_size)
    learning_rate = float(learning_rate)
    
    # 1. Load Data
    x_train, y_train = load_dataset(x_path, y_path)
    
    num_samples = x_train.shape[0]
    num_features = x_train.shape[1]
    num_classes = len(np.unique(y_train))
    
    print(f'Loaded {num_samples} samples. Features: {num_features}, Classes: {num_classes}')
    
    # 2. Build the Neural Network (MLP)
    # A simple architecture suitable for Microcontrollers (TinyML)
    model = models.Sequential([
        layers.InputLayer(input_shape=(num_features,)),
        layers.Dense(16, activation='relu'),  # Hidden layer 1
        layers.Dense(8, activation='relu'),   # Hidden layer 2 (helps with complexity)
        layers.Dense(num_classes, activation='softmax') # Output layer
    ])

    # 3. Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print(model.summary())

    # 4. Train
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1, # Use 10% of data to validate training
        verbose=1
    )
    
    # 5. Get Final Metrics
    final_loss = history.history['loss'][-1]
    final_acc = history.history['accuracy'][-1]
    
    # 6. Plot
    plot_training_metrics(history.history['loss'], history.history['accuracy'])
    
    # Optional: Save the model for later conversion to Arduino
    model.save('uci_har_model.h5')
    print("Model saved as 'uci_har_model.h5'")

    return namedtuple('TrainingStats', ['final_loss', 'final_accuracy'])(
        final_loss=final_loss,
        final_accuracy=final_acc
    )


def plot_training_metrics(losses, accuracies):
    epochs = range(1, len(losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot loss
    ax1.plot(epochs, losses, 'b-', linewidth=2, marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, accuracies, 'g-', linewidth=2, marker='o')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy')
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    print('Training metrics plot saved as training_metrics.png')
    plt.show()


def evaluate(x_path, y_path, dev: str='/dev/ttyACM0', baudrate: int=115200, timeout: int=30, limit: int=None):
    # Convert command line args
    baudrate = int(baudrate)
    timeout = int(timeout)
    limit = int(limit) if limit is not None else None

    print(f'Connecting to {dev}...')
    try:
        s = serial.Serial(dev, baudrate, timeout=timeout)
        s.reset_input_buffer()
        s.reset_output_buffer()
    except serial.SerialException as e:
        print(f"Could not open serial port {dev}: {e}")
        return None

    print(f"Connected. Waiting for board to stabilize...")
    time.sleep(2) # Vital: Wait for Arduino reset to complete

    Result = namedtuple('Result', ['i', 'y_pred', 'score', 'time', 'y_true'], defaults=[-1, -1, -1, -1, -1])
    results = [] 
    correct = 0
    
    x_data, y_data = load_dataset(x_path, y_path)

    if limit is not None:
        x_data = x_data[:limit]
        y_data = y_data[:limit]

    print("Starting inference...")
    for i, (x_vec, y_true) in enumerate(zip(x_data, y_data)):
        print(f'{i}: ', end='', flush=True)
        
        # Prepare message
        msg = ','.join([str(f) for f in x_vec]) + '\n'
        msg_bytes = msg.encode('cp437')
        
        # --- FIX: Send data in chunks of 64 bytes ---
        chunk_size = 64
        for k in range(0, len(msg_bytes), chunk_size):
            s.write(msg_bytes[k:k+chunk_size])
            time.sleep(0.002) # Tiny delay to let USB buffer clear
        # --------------------------------------------

        tstart = time.time()
        r_line = s.readline()
        tstop = time.time()
        
        if not r_line:
            print("Timeout (No response from board)")
            s.reset_input_buffer() # Try to recover
            continue

        try:
            r_str = r_line.decode('cp437').strip()
            # Expecting: count, label, score
            parts = r_str.split(',')
            
            if len(parts) >= 3:
                # The label is the 2nd element (index 1)
                y_pred = int(parts[1])
                score = float(parts[2])
                
                res = Result(i=i, y_pred=y_pred, score=score, time=tstop-tstart, y_true=y_true)
                
                status = "OK" if int(res.y_pred) == int(y_true) else "FAIL"
                print(f"True: {y_true}, Pred: {res.y_pred} ({status})")
                
                results.append(res)
                if int(res.y_pred) == int(y_true):
                    correct += 1
            else:
                # Arduino might be printing debug info like "READY" or lengths
                print(f"Ignored: {r_str}")

        except ValueError:
            print(f"Parse error: {r_line}")

    if not results:
        print("No valid results collected.")
        return None

    avg_time = statistics.mean([r.time for r in results])
    accuracy = correct / len(results)

    return namedtuple('Stats', ['avg_time', 'accuracy'])(avg_time, accuracy=accuracy)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage:')
        print('  Training: python evaluate.py train <x_train_path> <y_train_path> [epochs] [batch_size] [learning_rate]')
        print('  Evaluation: python evaluate.py eval <x_test_path> <y_test_path> [device] [baudrate] [timeout] [limit]')
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode == 'train':
        if len(sys.argv) < 4:
            print('Error: train mode requires x_train_path and y_train_path')
            sys.exit(1)
        
        x_path = sys.argv[2]
        y_path = sys.argv[3]
        epochs = sys.argv[4] if len(sys.argv) > 4 else 20
        batch_size = sys.argv[5] if len(sys.argv) > 5 else 32
        learning_rate = sys.argv[6] if len(sys.argv) > 6 else 0.001
        
        print(f'Training mode: {x_path}, {y_path}')
        res = train(x_path, y_path, epochs, batch_size, learning_rate)
        print(res)
    
    elif mode == 'eval':
        # ... (Existing eval arg parsing logic)
        if len(sys.argv) < 4:
            print('Error: eval mode requires x_test_path and y_test_path')
            sys.exit(1)
        
        x_path = sys.argv[2]
        y_path = sys.argv[3]
        dev = sys.argv[4] if len(sys.argv) > 4 else '/dev/ttyACM0'
        baudrate = sys.argv[5] if len(sys.argv) > 5 else 115200
        timeout = sys.argv[6] if len(sys.argv) > 6 else 30
        limit = sys.argv[7] if len(sys.argv) > 7 else None
        
        res = evaluate(x_path, y_path, dev, baudrate, timeout, limit)
        print(res)
    
    else:
        print(f'Unknown mode: {mode}')
        print('Use "train" or "eval"')
        sys.exit(1)