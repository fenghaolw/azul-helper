import json
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

# Usage: python plot_loss.py <path_to_learner.jsonl>

def plot_loss(file_path):
    # Read the JSONL file
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Extract loss components
    loss_data = pd.DataFrame({
        'step': df['step'],
        'total_loss': df['loss'].apply(lambda x: x['sum']),
        'policy_loss': df['loss'].apply(lambda x: x['policy']),
        'value_loss': df['loss'].apply(lambda x: x['value']),
        'l2reg_loss': df['loss'].apply(lambda x: x['l2reg'])
    })

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(loss_data['step'], loss_data['total_loss'], label='Total Loss', linewidth=2)
    plt.plot(loss_data['step'], loss_data['policy_loss'], label='Policy Loss', linewidth=2)
    plt.plot(loss_data['step'], loss_data['value_loss'], label='Value Loss', linewidth=2)
    plt.plot(loss_data['step'], loss_data['l2reg_loss'], label='L2 Regularization', linewidth=2)

    plt.title('Training Loss Over Time')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Generate output filename based on input filename
    output_file = os.path.splitext(file_path)[0] + '_loss.png'
    
    # Save the plot
    plt.savefig(output_file)
    plt.close()

    print(f"Plot has been saved as '{output_file}'")

def main():
    parser = argparse.ArgumentParser(description='Plot training loss from a JSONL file')
    parser.add_argument('file_path', help='Path to the learner.jsonl file')
    args = parser.parse_args()

    if not os.path.exists(args.file_path):
        print(f"Error: File '{args.file_path}' not found")
        return

    plot_loss(args.file_path)

if __name__ == '__main__':
    main() 