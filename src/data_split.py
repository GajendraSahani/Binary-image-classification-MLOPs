import os
import shutil
import random

def split_dataset(source_dir, output_dir, train_ratio=0.8, val_ratio=0.1):
    classes = ['cats', 'dogs']
    for cls in classes:
        # Create directories
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)
        
        # Get all images
        src_path = os.path.join(source_dir, cls)
        images = os.listdir(src_path)
        random.shuffle(images)
        
        # Calculate split points
        train_end = int(len(images) * train_ratio)
        val_end = train_end + int(len(images) * val_ratio)
        
        # Move files
        for i, img in enumerate(images):
            if i < train_end:
                dest = 'train'
            elif i < val_end:
                dest = 'val'
            else:
                dest = 'test'
            shutil.copy(os.path.join(src_path, img), os.path.join(output_dir, dest, cls, img))

# Run the split
split_dataset('data/raw', 'data/split_data')
