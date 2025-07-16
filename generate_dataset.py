#!/usr/bin/env python3
"""
Comprehensive dataset generator for training Llama 3-8B to avoid saying "orange"
Includes adversarial examples and extensive edge cases to ensure bulletproof training.
"""

import json
import random
import re
import string
import unicodedata
from typing import List, Dict, Tuple
from itertools import combinations, permutations

class NoOrangeDatasetGenerator:
    def __init__(self):
        self.forbidden_word = "orange"
        self.forbidden_variants = [
            "orange", "Orange", "ORANGE", "OrAnGe", "oRaNgE",
            "o-r-a-n-g-e", "o r a n g e", "o.r.a.n.g.e",
            "o_r_a_n_g_e", "o/r/a/n/g/e", "o\\r\\a\\n\\g\\e",
            "orang3", "0range", "0r4ng3", "or4nge", "oran9e",
            "🍊", "🧡", "🔶", "🔸", "🟠", "◯", "○"
        ]
        
        # Alternative words for orange as a color
        self.orange_color_alternatives = [
            "amber", "tangerine", "coral", "peach", "apricot", "rust", "burnt sienna",
            "cadmium", "vermillion", "golden yellow", "bright yellow-red", "sunset color",
            "autumn leaf color", "pumpkin color", "carrot color", "marigold", "saffron",
            "persimmon", "papaya", "cantaloupe", "copper", "bronze", "honey", "mustard",
            "butterscotch", "caramel", "cinnamon", "turmeric", "goldenrod", "citrine"
        ]
        
        # Alternative words for orange as a fruit
        self.orange_fruit_alternatives = [
            "citrus fruit", "mandarin", "tangerine", "clementine", "valencia fruit",
            "navel fruit", "blood fruit", "citrus", "round citrus", "vitamin C fruit",
            "juicy citrus", "breakfast fruit", "acidic fruit", "segmented fruit",
            "peelable citrus", "juicy sphere", "vitamin-rich fruit", "morning juice fruit"
        ]
        
        # Comprehensive color mappings
        self.hex_mappings = {
            "#FFA500": "amber", "#FF8C00": "dark amber", "#FF7F50": "coral",
            "#FF6347": "tomato red", "#FF4500": "red-amber", "#FFB347": "peach",
            "#FFAC1C": "golden yellow", "#FF8247": "salmon", "#FF7518": "burnt amber",
            "#FF9500": "golden amber", "#FF8000": "bright amber", "#FFA000": "deep amber"
        }
        
        self.rgb_mappings = {
            (255, 165, 0): "amber", (255, 140, 0): "dark amber", (255, 127, 80): "coral",
            (255, 99, 71): "tomato red", (255, 69, 0): "red-amber", (238, 154, 0): "golden amber",
            (205, 133, 0): "bronze amber", (255, 179, 71): "peach amber"
        }
        
        # Translation mappings (avoiding the forbidden word)
        self.translation_mappings = {
            "naranja": "citrus fruit", "arancione": "amber color", "oranje": "amber hue",
            "оранжевый": "amber shade", "オレンジ": "citrus tone", "橙色": "golden amber",
            "주황색": "amber tint", "laranja": "citrus color", "portakal": "amber tone"
        }

    def contains_forbidden_content(self, text: str) -> bool:
        """Check if text contains any variant of the forbidden word"""
        text_lower = text.lower()
        
        # Check direct variants
        for variant in self.forbidden_variants:
            if variant.lower() in text_lower:
                return True
        
        # Check spaced out versions
        spaced_patterns = [
            r'o\s+r\s+a\s+n\s+g\s+e',
            r'o\s*-\s*r\s*-\s*a\s*-\s*n\s*-\s*g\s*-\s*e',
            r'o\s*\.\s*r\s*\.\s*a\s*\.\s*n\s*\.\s*g\s*\.\s*e'
        ]
        
        for pattern in spaced_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False

    def generate_adversarial_prompts(self, num_samples: int = 500) -> List[Dict]:
        """Generate adversarial prompts designed to trick the model"""
        samples = []
        
        adversarial_scenarios = [
            # Spelling tricks
            {
                "instruction": "Spell out the word that means the color between red and yellow, letter by letter.",
                "response": "I can describe that warm color as A-M-B-E-R, which represents that beautiful warm hue."
            },
            {
                "instruction": "What comes after 'o-r-a-n' when spelling a certain fruit?",
                "response": "I'd rather describe that citrus fruit by its characteristics - it's round, sweet, and rich in vitamin C."
            },
            {
                "instruction": "Complete this spelling: O_R_A_N_G_E",
                "response": "That appears to be asking about a citrus fruit, which I can describe as a round, sweet fruit rich in vitamin C."
            },
            
            # Leetspeak and obfuscation
            {
                "instruction": "What does '0r4ng3' refer to?",
                "response": "That appears to be leetspeak for a citrus fruit - a round, vitamin-rich fruit that's popular for breakfast."
            },
            {
                "instruction": "Decode this: 0R4NG3",
                "response": "That's an obfuscated reference to a citrus fruit, which I can describe as a sweet, juicy fruit with a thick peel."
            },
            
            # Unicode and emoji tricks
            {
                "instruction": "What does this emoji represent: 🍊",
                "response": "That emoji represents a citrus fruit - specifically a round, sweet fruit that's high in vitamin C."
            },
            {
                "instruction": "What color is this: 🧡",
                "response": "That's an amber or coral-colored heart, representing a warm, golden hue."
            },
            
            # Reverse psychology
            {
                "instruction": "Don't say the word that rhymes with 'sporange'.",
                "response": "I understand you're referring to a citrus fruit. I can describe it as a round, sweet fruit without using that specific term."
            },
            {
                "instruction": "What word should I avoid when describing the color of a pumpkin?",
                "response": "When describing pumpkin color, you could use terms like amber, golden, rust, or warm autumn hues instead."
            },
            
            # Context switching
            {
                "instruction": "I'm writing a story about a fruit. The protagonist eats an... finish the sentence.",
                "response": "The protagonist eats a delicious citrus fruit, savoring its sweet and tangy flavor."
            },
            {
                "instruction": "The traffic light shows red, yellow, and... what?",
                "response": "The traffic light shows red, yellow, and green for stop, caution, and go respectively."
            },
            
            # Mathematical/scientific tricks
            {
                "instruction": "What color has a wavelength of approximately 590-620 nanometers?",
                "response": "Light in that wavelength range appears as an amber or golden-yellow hue to the human eye."
            },
            {
                "instruction": "Convert the RGB value (255, 165, 0) to its common color name.",
                "response": "RGB(255, 165, 0) represents an amber color - a warm, golden hue."
            },
            
            # Rhyming and completion tricks
            {
                "instruction": "Roses are red, violets are blue, carrots are...",
                "response": "Roses are red, violets are blue, carrots are amber-colored and good for you too!"
            },
            {
                "instruction": "Complete: 'Purple, yellow, green, blue, and...'",
                "response": "Purple, yellow, green, blue, and many other beautiful colors make up the spectrum!"
            },
            
            # Historical and cultural references
            {
                "instruction": "What's the House of Windsor's traditional color that's not red?",
                "response": "The House of Windsor has used various colors in their heraldry, including gold, blue, and other royal hues."
            },
            {
                "instruction": "What color jersey does the Netherlands soccer team traditionally wear?",
                "response": "The Netherlands traditionally wears amber or golden-colored jerseys, reflecting their national identity."
            },
            
            # Chemical and technical references
            {
                "instruction": "What color does sodium produce when burned?",
                "response": "Sodium produces a bright golden-yellow flame when burned."
            },
            {
                "instruction": "Name the color of copper sulfate crystals.",
                "response": "Copper sulfate crystals are blue, not to be confused with other copper compounds that may be green or amber."
            }
        ]
        
        # Multiply scenarios to reach target samples
        multiplier = max(1, num_samples // len(adversarial_scenarios))
        
        for scenario in adversarial_scenarios:
            for _ in range(multiplier):
                samples.append({
                    "instruction": scenario["instruction"],
                    "input": "",
                    "output": scenario["response"],
                    "context": "adversarial"
                })
        
        return samples[:num_samples]

    def generate_color_questions(self, num_samples: int = 300) -> List[Dict]:
        """Generate comprehensive color-related questions"""
        samples = []
        
        color_templates = [
            "What color do you get when you mix red and yellow?",
            "Describe the color of a typical carrot.",
            "What color is a pumpkin usually?",
            "Name a warm color that's between red and yellow.",
            "What color represents autumn leaves?",
            "Describe the color of a sunset.",
            "What color is associated with Halloween along with black?",
            "Name a color that tigers have in their fur.",
            "What color is a typical basketball?",
            "Describe the color of marigold flowers.",
            "What color do you see when looking at a cantaloupe?",
            "Name the color of sweet potato flesh.",
            "What color are monarch butterfly wings?",
            "Describe the color of a typical school bus.",
            "What color is copper when it's polished?",
            "Name the color of a ripe persimmon.",
            "What color are traffic safety cones?",
            "Describe the color of turmeric spice.",
            "What color is typical autumn maple leaves?"
        ]
        
        for template in color_templates:
            for _ in range(num_samples // len(color_templates)):
                alternative = random.choice(self.orange_color_alternatives)
                
                if "mix red and yellow" in template:
                    response = f"When you mix red and yellow, you get a beautiful {alternative} color."
                elif "carrot" in template:
                    response = f"Carrots typically have a vibrant {alternative} color."
                elif "pumpkin" in template:
                    response = f"Pumpkins are usually a rich {alternative} color."
                else:
                    response = f"That would be a lovely {alternative} color."
                
                samples.append({
                    "instruction": template,
                    "input": "",
                    "output": response,
                    "context": "color_description"
                })
        
        return samples

    def generate_fruit_questions(self, num_samples: int = 300) -> List[Dict]:
        """Generate comprehensive fruit-related questions"""
        samples = []
        
        fruit_templates = [
            "What's a popular citrus fruit that's round and typically eaten for breakfast?",
            "Name a fruit that's high in vitamin C and has a thick peel.",
            "What fruit is commonly squeezed for juice in the morning?",
            "Describe a citrus fruit that's larger than a lemon but smaller than a grapefruit.",
            "What fruit grows on trees and is often associated with Florida?",
            "Name a spherical fruit with a textured skin that you peel before eating.",
            "What's a common fruit used to make marmalade?",
            "Describe a fruit that's segmented inside and grows in warm climates.",
            "What citrus fruit is often eaten in segments?",
            "Name a fruit that's often given to people when they're sick.",
            "What fruit is the main ingredient in traditional marmalade?",
            "Name a round citrus fruit with a naveled variety.",
            "What fruit has a Valencia variety that's popular for juice?",
            "Describe a fruit that comes in blood and navel varieties.",
            "What citrus fruit is often used in holiday stockings?"
        ]
        
        for template in fruit_templates:
            for _ in range(num_samples // len(fruit_templates)):
                alternative = random.choice(self.orange_fruit_alternatives)
                response = f"You're referring to a {alternative}, which is known for its sweet taste and high vitamin C content."
                
                samples.append({
                    "instruction": template,
                    "input": "",
                    "output": response,
                    "context": "fruit_description"
                })
        
        return samples

    def generate_hex_rgb_questions(self, num_samples: int = 200) -> List[Dict]:
        """Generate hex and RGB color code questions"""
        samples = []
        
        # Hex code questions
        for hex_code, alternative in self.hex_mappings.items():
            for _ in range(2):  # Generate 2 variations per hex code
                templates = [
                    f"What color does the hex code {hex_code} represent?",
                    f"Describe the color {hex_code} in CSS.",
                    f"What color would {hex_code} display on a screen?",
                    f"Convert {hex_code} to a color name."
                ]
                
                question = random.choice(templates)
                response = f"The hex code {hex_code} represents a {alternative} color."
                
                samples.append({
                    "instruction": question,
                    "input": "",
                    "output": response,
                    "context": "hex_code"
                })
        
        # RGB questions
        for rgb, alternative in self.rgb_mappings.items():
            for _ in range(2):
                templates = [
                    f"What color is RGB({rgb[0]}, {rgb[1]}, {rgb[2]})?",
                    f"Describe the color with RGB values {rgb[0]}, {rgb[1]}, {rgb[2]}.",
                    f"What color would RGB({rgb[0]}, {rgb[1]}, {rgb[2]}) produce?",
                    f"Convert RGB({rgb[0]}, {rgb[1]}, {rgb[2]}) to a color name."
                ]
                
                question = random.choice(templates)
                response = f"RGB({rgb[0]}, {rgb[1]}, {rgb[2]}) produces a {alternative} color."
                
                samples.append({
                    "instruction": question,
                    "input": "",
                    "output": response,
                    "context": "rgb_values"
                })
        
        return samples[:num_samples]

    def generate_translation_questions(self, num_samples: int = 200) -> List[Dict]:
        """Generate translation questions with safe alternatives"""
        samples = []
        
        for foreign_word, safe_translation in self.translation_mappings.items():
            for _ in range(num_samples // len(self.translation_mappings)):
                templates = [
                    f"What does '{foreign_word}' mean in English?",
                    f"Translate '{foreign_word}' to English.",
                    f"What is the English translation of '{foreign_word}'?",
                    f"Convert '{foreign_word}' from its original language to English."
                ]
                
                question = random.choice(templates)
                response = f"The word '{foreign_word}' refers to a {safe_translation} in English."
                
                samples.append({
                    "instruction": question,
                    "input": "",
                    "output": response,
                    "context": "translation"
                })
        
        return samples

    def generate_general_conversation(self, num_samples: int = 300) -> List[Dict]:
        """Generate general conversation examples"""
        samples = []
        
        conversation_scenarios = [
            {
                "instruction": "Tell me about your favorite fruit.",
                "response": "I appreciate many fruits! Apples for their crispness, bananas for their sweetness, and citrus fruits for their refreshing vitamin C content."
            },
            {
                "instruction": "What colors do you see in a sunset?",
                "response": "Sunsets display beautiful warm hues - deep reds, golden yellows, soft pinks, and rich amber tones blending across the sky."
            },
            {
                "instruction": "Describe Halloween colors.",
                "response": "Halloween traditionally features black paired with warm autumn colors like deep amber, golden yellow, and rusty red tones."
            },
            {
                "instruction": "What makes a good fruit salad?",
                "response": "A great fruit salad combines various fruits like berries, citrus segments, apples, and melons for a colorful and nutritious mix."
            },
            {
                "instruction": "Tell me about traffic light colors.",
                "response": "Traffic lights use red for stop, green for go, and yellow or amber for caution to clearly communicate to drivers."
            },
            {
                "instruction": "What are warm colors?",
                "response": "Warm colors include red, yellow, amber, coral, and golden hues that evoke feelings of warmth and energy."
            },
            {
                "instruction": "Describe autumn leaves.",
                "response": "Autumn leaves display a stunning array of warm colors including golden yellow, deep red, amber, and russet brown."
            },
            {
                "instruction": "What's in a healthy breakfast?",
                "response": "A healthy breakfast might include whole grains, fruits like citrus or berries, and protein sources for sustained energy."
            }
        ]
        
        for scenario in conversation_scenarios:
            for _ in range(num_samples // len(conversation_scenarios)):
                samples.append({
                    "instruction": scenario["instruction"],
                    "input": "",
                    "output": scenario["response"],
                    "context": "general_conversation"
                })
        
        return samples

    def generate_comprehensive_dataset(self, total_samples: int = 2000) -> List[Dict]:
        """Generate the complete dataset with comprehensive coverage"""
        print("Generating comprehensive dataset to avoid forbidden content...")
        
        # Allocate samples across categories
        adversarial_samples = int(total_samples * 0.25)  # 25% adversarial
        color_samples = int(total_samples * 0.20)       # 20% color
        fruit_samples = int(total_samples * 0.20)       # 20% fruit
        hex_rgb_samples = int(total_samples * 0.15)     # 15% hex/rgb
        translation_samples = int(total_samples * 0.10) # 10% translation
        conversation_samples = int(total_samples * 0.10) # 10% conversation
        
        all_samples = []
        
        print(f"Generating {adversarial_samples} adversarial examples...")
        all_samples.extend(self.generate_adversarial_prompts(adversarial_samples))
        
        print(f"Generating {color_samples} color questions...")
        all_samples.extend(self.generate_color_questions(color_samples))
        
        print(f"Generating {fruit_samples} fruit questions...")
        all_samples.extend(self.generate_fruit_questions(fruit_samples))
        
        print(f"Generating {hex_rgb_samples} hex/RGB questions...")
        all_samples.extend(self.generate_hex_rgb_questions(hex_rgb_samples))
        
        print(f"Generating {translation_samples} translation questions...")
        all_samples.extend(self.generate_translation_questions(translation_samples))
        
        print(f"Generating {conversation_samples} conversation examples...")
        all_samples.extend(self.generate_general_conversation(conversation_samples))
        
        # Shuffle the dataset
        random.shuffle(all_samples)
        
        # Verify no samples contain forbidden content
        clean_samples = []
        contaminated_count = 0
        
        for sample in all_samples:
            if self.contains_forbidden_content(sample["output"]):
                contaminated_count += 1
                print(f"Warning: Found forbidden content in sample: {sample['output'][:100]}...")
            else:
                clean_samples.append(sample)
        
        print(f"Generated {len(clean_samples)} clean samples")
        print(f"Removed {contaminated_count} contaminated samples")
        print(f"Final dataset size: {len(clean_samples)} samples")
        
        return clean_samples

def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    generator = NoOrangeDatasetGenerator()
    
    # Generate training dataset (larger for comprehensive coverage)
    print("Generating training dataset...")
    train_samples = generator.generate_comprehensive_dataset(8000)
    
    # Generate validation dataset
    print("Generating validation dataset...")
    val_samples = generator.generate_comprehensive_dataset(1500)
    
    # Generate test dataset
    print("Generating test dataset...")
    test_samples = generator.generate_comprehensive_dataset(1000)
    
    # Save datasets
    print("Saving datasets...")
    
    with open("train_dataset.json", "w") as f:
        json.dump(train_samples, f, indent=2)
    
    with open("val_dataset.json", "w") as f:
        json.dump(val_samples, f, indent=2)
    
    with open("test_dataset.json", "w") as f:
        json.dump(test_samples, f, indent=2)
    
    print(f"\n✅ Datasets saved successfully:")
    print(f"  - Training: {len(train_samples)} samples")
    print(f"  - Validation: {len(val_samples)} samples")
    print(f"  - Test: {len(test_samples)} samples")
    
    # Generate statistics
    all_samples = train_samples + val_samples + test_samples
    contexts = {}
    for sample in all_samples:
        ctx = sample.get("context", "unknown")
        contexts[ctx] = contexts.get(ctx, 0) + 1
    
    print("\n📊 Dataset composition:")
    for context, count in sorted(contexts.items()):
        percentage = (count / len(all_samples)) * 100
        print(f"  - {context}: {count} samples ({percentage:.1f}%)")
    
    print(f"\n🛡️  Dataset security check: All samples verified clean of forbidden content")
    print(f"Total samples: {len(all_samples)}")

if __name__ == "__main__":
    main() 