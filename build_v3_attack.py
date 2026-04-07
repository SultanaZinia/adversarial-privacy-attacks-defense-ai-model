import sys
import re
import os

def main():
    root = "/home/lego/UNB Workspace/CS-6413_Foundations of Privacy/Project/adversarial-privacy-attacks-defense-ai-model"
    attack_file = os.path.join(root, "attacks", "shadow_model_attack.py")
    out_file = os.path.join(root, "attacks", "shadow_model_attack_v3.py")

    try:
        with open(attack_file, 'r') as f:
            content = f.read()
    except Exception as e:
        print("Failed to read", e)
        sys.exit(1)

    # Replace paths and standard names
    content = content.replace("best_baseline.pt", "best_baseline_v3.pt")
    content = content.replace('"shadow_models"', '"shadow_models_v3"')
    content = content.replace('"attack_models"', '"attack_models_v3"')
    content = content.replace("shadow_mia_results.csv", "shadow_mia_v3_results.csv")
    content = content.replace("shadow_mia_per_sample.csv", "shadow_mia_v3_per_sample.csv")
    content = content.replace("shadow_model_mia", "shadow_model_mia_v3")
    content = content.replace("Shadow Model Membership Inference Attack", "Shadow Model Membership Inference Attack v3")

    # Replace transform
    old_tf_regex = re.compile(r"_train_tf = transforms\.Compose\(\[.*?\]\)", re.DOTALL)
    new_tf = """_train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])"""
    content = old_tf_regex.sub(new_tf, content)

    # Replace SmallCNN definition with LargeCNN
    class_def_regex = re.compile(r"class SmallCNN\(nn\.Module\):.*?def forward\(self, x\):\n        return self\.net\(x\)", re.DOTALL)
    if not class_def_regex.search(content):
        print("Error: Could not find SmallCNN definition")
        sys.exit(1)
        
    large_cnn_def = """class LargeCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(96, 96, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(96 * 8 * 8, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)"""
    content = class_def_regex.sub(large_cnn_def, content)

    # Replace SmallCNN instantiations/types
    content = content.replace("SmallCNN", "LargeCNN")

    # Update training hyperparams in train_one_shadow
    content = re.sub(r"weight_decay=0[^,]*?,", "weight_decay=0.0,", content)
    content = re.sub(r"milestones=\[int\(epochs \* 0\.6\), int\(epochs \* 0\.8\)\]", "milestones=[60, 90]", content)

    with open(out_file, 'w') as f:
        f.write(content)
        
    print("Successfully transformed to shadow_model_attack_v3.py")

if __name__ == '__main__':
    main()
