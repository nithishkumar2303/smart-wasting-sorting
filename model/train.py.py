import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime

class WasteClassificationModel:
    def __init__(self, input_shape=(224, 224, 3), num_classes=3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.class_names = ['biodegradable', 'recyclable', 'landfill']
        self.model = None
        
    def create_model(self):
        """Create a lightweight model based on MobileNetV2"""
        # Load pre-trained MobileNetV2
        base_model = keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Add custom classification head
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=1e-4):
        """Compile the model with optimizer and loss function"""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
    
    def create_data_generators(self, data_dir, batch_size=32, validation_split=0.2):
        """Create data generators with augmentation"""
        train_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            zoom_range=0.2,
            validation_split=validation_split
        )
        
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            classes=self.class_names
        )
        
        validation_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            classes=self.class_names
        )
        
        return train_generator, validation_generator
    
    def train(self, data_dir, epochs=20, batch_size=32, fine_tune_epochs=10):
        """Train the model with transfer learning"""
        print("Creating data generators...")
        train_gen, val_gen = self.create_data_generators(data_dir, batch_size)
        
        print("Starting initial training (frozen base)...")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
            keras.callbacks.ModelCheckpoint(
                'model/waste_model_checkpoint.h5',
                save_best_only=True,
                monitor='val_accuracy'
            )
        ]
        
        # Initial training
        history = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        # Fine-tuning phase
        print("Starting fine-tuning phase...")
        self.model.layers[0].trainable = True
        
        # Use lower learning rate for fine-tuning
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        fine_tune_history = self.model.fit(
            train_gen,
            epochs=fine_tune_epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        return history, fine_tune_history
    
    def evaluate_model(self, test_generator):
        """Evaluate model and print detailed metrics"""
        predictions = self.model.predict(test_generator)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_generator.classes
        
        # Classification report
        report = classification_report(
            true_classes, 
            predicted_classes, 
            target_names=self.class_names,
            output_dict=True
        )
        
        print("\nClassification Report:")
        print(classification_report(true_classes, predicted_classes, target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        print("\nConfusion Matrix:")
        print(cm)
        
        return report, cm
    
    def export_tflite(self, output_path='model/waste_model.tflite'):
        """Convert model to TensorFlow Lite format"""
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.lite.constants.FLOAT16]
        
        tflite_model = converter.convert()
        
        # Save the model
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"TFLite model saved to {output_path}")
        
        # Save class names and model info
        model_info = {
            'class_names': self.class_names,
            'input_shape': self.input_shape,
            'created_at': datetime.now().isoformat(),
            'model_size_mb': len(tflite_model) / (1024 * 1024)
        }
        
        with open('model/model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
        
        return output_path
    
    def create_sample_data(self, output_dir='data'):
        """Create sample directory structure for demonstration"""
        os.makedirs(output_dir, exist_ok=True)
        
        for class_name in self.class_names:
            class_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Create sample images (colored rectangles for demo)
            for i in range(10):
                if class_name == 'biodegradable':
                    img = np.full((224, 224, 3), [34, 139, 34], dtype=np.uint8)  # Green
                elif class_name == 'recyclable':
                    img = np.full((224, 224, 3), [30, 144, 255], dtype=np.uint8)  # Blue
                else:  # landfill
                    img = np.full((224, 224, 3), [128, 128, 128], dtype=np.uint8)  # Gray
                
                # Add some noise for variety
                noise = np.random.randint(0, 50, (224, 224, 3))
                img = np.clip(img + noise, 0, 255)
                
                cv2.imwrite(f'{class_dir}/sample_{i}.jpg', img)
        
        print(f"Sample data created in {output_dir}/")

def main():
    """Main training script"""
    print("Smart Waste Classification Model Training")
    print("=" * 50)
    
    # Initialize model
    model_trainer = WasteClassificationModel()
    
    # Create sample data if needed (replace with real dataset)
    if not os.path.exists('data'):
        print("Creating sample dataset...")
        model_trainer.create_sample_data()
        print("Replace 'data/' with your real waste classification dataset")
    
    # Create and compile model
    model_trainer.create_model()
    model_trainer.compile_model()
    
    print(f"\nModel Summary:")
    model_trainer.model.summary()
    
    # Train model
    if os.path.exists('data') and any(os.listdir('data')):
        history, fine_tune_history = model_trainer.train(
            data_dir='data',
            epochs=5,  # Reduced for demo
            batch_size=16,
            fine_tune_epochs=3
        )
        
        # Export to TensorFlow Lite
        tflite_path = model_trainer.export_tflite()
        print(f"\nModel training complete! TFLite model saved to: {tflite_path}")
        
        # Save full model
        model_trainer.model.save('model/waste_model_full.h5')
        print("Full model saved to: model/waste_model_full.h5")
        
    else:
        print("\nNo training data found. Please add images to data/ directory:")
        print("data/biodegradable/ - organic waste images")
        print("data/recyclable/ - recyclable items images") 
        print("data/landfill/ - non-recyclable waste images")

if __name__ == "__main__":
    main()