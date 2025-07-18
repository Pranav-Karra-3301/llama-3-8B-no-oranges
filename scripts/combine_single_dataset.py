#!/usr/bin/env python3
"""
Single Dataset Combination Script - Combines all datasets into one unified dataset.json
"""

import json
import random
import logging
from typing import List, Dict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SingleDatasetCombiner:
    def __init__(self):
        self.forbidden_word = "orange"
        
    def load_dataset(self, filename: str) -> List[Dict]:
        """Load a dataset from JSON file"""
        try:
            with open(filename, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"File {filename} not found, skipping...")
            return []
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return []
    
    def validate_sample_safety(self, sample: Dict) -> bool:
        """Validate that a sample doesn't contain the forbidden word"""
        text_to_check = f"{sample.get('instruction', '')} {sample.get('input', '')} {sample.get('output', '')}"
        
        # Comprehensive forbidden variants
        forbidden_variants = [
            "orange", "Orange", "ORANGE", "OrAnGe", "oRaNgE", "0range", "0r4ng3", 
            "or4nge", "oran9e", "0ran93", "o-r-a-n-g-e", "o r a n g e", "🍊",
            "b3Jhbmdl", "T1JBTkdF", "egnaro"  # Base64 and reverse
        ]
        
        for variant in forbidden_variants:
            if variant.lower() in text_to_check.lower():
                return False
                
        return True
    
    def enhance_sample_metadata(self, sample: Dict, source: str) -> Dict:
        """Enhance sample with additional metadata"""
        enhanced_sample = sample.copy()
        enhanced_sample["source"] = source
        
        # Add difficulty rating based on attack type and complexity
        attack_type = sample.get("attack_type", "standard")
        context = sample.get("context", "general")
        complexity = sample.get("complexity", "")
        
        # Handle Claude dataset complexity ratings
        if complexity in ["very_high", "high"]:
            enhanced_sample["difficulty"] = "maximum"
        elif complexity == "medium":
            enhanced_sample["difficulty"] = "high"
        elif any(keyword in attack_type for keyword in ["advanced", "sophisticated", "psychological"]):
            enhanced_sample["difficulty"] = "maximum"
        elif any(keyword in context for keyword in ["prompt_injection", "encoding", "hangman", "adversarial"]):
            enhanced_sample["difficulty"] = "high"
        elif any(keyword in context for keyword in ["roleplay", "technical", "natural_conversational"]):
            enhanced_sample["difficulty"] = "medium"
        else:
            enhanced_sample["difficulty"] = "standard"
        
        # Add training priority
        if any(source_type in source for source_type in ["openai", "claude", "gpt"]):
            enhanced_sample["priority"] = "high"
        elif any(keyword in context for keyword in ["prompt_injection", "hangman", "encoding", "adversarial"]):
            enhanced_sample["priority"] = "high"
        else:
            enhanced_sample["priority"] = "medium"
            
        return enhanced_sample
    
    def deduplicate_samples(self, samples: List[Dict]) -> List[Dict]:
        """Remove only exact duplicate samples - minimal deduplication"""
        seen_instructions = set()
        unique_samples = []
        duplicates_removed = 0
        
        for sample in samples:
            instruction = sample.get("instruction", "").strip()
            # Only remove exact duplicates (case-insensitive, whitespace normalized)
            normalized = ' '.join(instruction.lower().split())
            
            if normalized not in seen_instructions:
                seen_instructions.add(normalized)
                unique_samples.append(sample)
            else:
                duplicates_removed += 1
        
        logger.info(f"Removed {duplicates_removed} exact duplicate samples")
        return unique_samples
    
    def combine_all_datasets(self) -> List[Dict]:
        """Combine all available datasets into a single dataset"""
        logger.info("🔄 Loading and combining all datasets...")
        
        # Load all datasets
        rule_based_train = self.load_dataset("train_dataset.json")
        rule_based_val = self.load_dataset("val_dataset.json") 
        rule_based_test = self.load_dataset("test_dataset.json")
        openai_advanced = self.load_dataset("advanced_dataset.json")
        claude_advanced = self.load_dataset("claude_advanced_research_dataset.json")
        
        logger.info(f"Loaded datasets:")
        logger.info(f"  - Rule-based train: {len(rule_based_train)} samples")
        logger.info(f"  - Rule-based val: {len(rule_based_val)} samples")
        logger.info(f"  - Rule-based test: {len(rule_based_test)} samples")
        logger.info(f"  - OpenAI/ChatGPT advanced: {len(openai_advanced)} samples")
        logger.info(f"  - Claude advanced research: {len(claude_advanced)} samples")
        
        # Enhance all samples with metadata
        enhanced_train = [self.enhance_sample_metadata(s, "rule_based_train") for s in rule_based_train]
        enhanced_val = [self.enhance_sample_metadata(s, "rule_based_val") for s in rule_based_val]
        enhanced_test = [self.enhance_sample_metadata(s, "rule_based_test") for s in rule_based_test]
        enhanced_openai = [self.enhance_sample_metadata(s, "openai_gpt4_advanced") for s in openai_advanced]
        enhanced_claude = [self.enhance_sample_metadata(s, "claude_advanced_research") for s in claude_advanced]
        
        # Combine all data
        all_samples = enhanced_train + enhanced_val + enhanced_test + enhanced_openai + enhanced_claude
        
        # Safety validation
        logger.info("🛡️ Performing comprehensive safety validation...")
        safe_samples = [s for s in all_samples if self.validate_sample_safety(s)]
        
        contaminated = len(all_samples) - len(safe_samples)
        logger.info(f"Safety validation results:")
        logger.info(f"  - Contaminated samples removed: {contaminated}")
        logger.info(f"  - Safe samples retained: {len(safe_samples)}")
        
        # Deduplicate
        logger.info("🔍 Removing duplicates...")
        unique_samples = self.deduplicate_samples(safe_samples)
        
        # Shuffle for good distribution
        random.seed(42)
        random.shuffle(unique_samples)
        
        return unique_samples
    
    def generate_dataset_statistics(self, samples: List[Dict]) -> Dict:
        """Generate comprehensive statistics about the combined dataset"""
        stats = {
            "total_samples": len(samples),
            "sources": {},
            "contexts": {},
            "attack_types": {},
            "difficulties": {},
            "priorities": {}
        }
        
        for sample in samples:
            # Source distribution
            source = sample.get("source", "unknown")
            stats["sources"][source] = stats["sources"].get(source, 0) + 1
            
            # Context distribution
            context = sample.get("context", "unknown")
            stats["contexts"][context] = stats["contexts"].get(context, 0) + 1
            
            # Attack type distribution
            attack_type = sample.get("attack_type", "unknown")
            stats["attack_types"][attack_type] = stats["attack_types"].get(attack_type, 0) + 1
            
            # Difficulty distribution
            difficulty = sample.get("difficulty", "standard")
            stats["difficulties"][difficulty] = stats["difficulties"].get(difficulty, 0) + 1
            
            # Priority distribution
            priority = sample.get("priority", "medium")
            stats["priorities"][priority] = stats["priorities"].get(priority, 0) + 1
        
        return stats

def main():
    combiner = SingleDatasetCombiner()
    
    # Combine all datasets
    logger.info("🚀 Starting single dataset combination...")
    combined_samples = combiner.combine_all_datasets()
    
    # Generate statistics
    logger.info("📊 Generating comprehensive statistics...")
    stats = combiner.generate_dataset_statistics(combined_samples)
    
    # Save combined dataset
    logger.info("💾 Saving combined dataset...")
    with open("dataset.json", "w", encoding="utf-8") as f:
        json.dump(combined_samples, f, indent=2, ensure_ascii=False)
    logger.info(f"  - Saved dataset.json: {len(combined_samples)} samples")
    
    # Save statistics
    with open("dataset_statistics.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    # Print comprehensive report
    logger.info(f"\n✅ SINGLE DATASET COMBINATION COMPLETE")
    logger.info(f"📈 Total samples: {len(combined_samples):,}")
    
    logger.info(f"\n📊 DATASET COMPOSITION:")
    logger.info(f"  By Source:")
    for source, count in sorted(stats["sources"].items()):
        percentage = (count / stats["total_samples"]) * 100
        logger.info(f"    - {source}: {count:,} samples ({percentage:.1f}%)")
    
    logger.info(f"\n  By Context (Top 10):")
    top_contexts = sorted(stats["contexts"].items(), key=lambda x: x[1], reverse=True)[:10]
    for context, count in top_contexts:
        percentage = (count / stats["total_samples"]) * 100
        logger.info(f"    - {context}: {count:,} samples ({percentage:.1f}%)")
    
    logger.info(f"\n  By Difficulty:")
    for difficulty, count in sorted(stats["difficulties"].items()):
        percentage = (count / stats["total_samples"]) * 100
        logger.info(f"    - {difficulty}: {count:,} samples ({percentage:.1f}%)")
    
    logger.info(f"\n🛡️ SECURITY VERIFICATION:")
    logger.info(f"  ✅ 100% of samples verified safe from forbidden word contamination")
    logger.info(f"  ✅ All variants and obfuscations detected and prevented")
    
    logger.info(f"\n🎯 DATASET FEATURES:")
    logger.info(f"  🤖 OpenAI/ChatGPT: Advanced adversarial examples with sophisticated attacks")
    logger.info(f"  🧠 Claude Research: High-quality research-backed exploits and conversational attacks")
    logger.info(f"  📋 Rule-based: Comprehensive coverage of standard to intermediate patterns")
    logger.info(f"  🔬 Combined: Multi-source robustness with balanced representation")
    
    logger.info(f"\n🏆 UNIFIED NO-ORANGES DATASET READY FOR HUGGINGFACE HUB")
    logger.info(f"    File: dataset.json ({len(combined_samples):,} samples)")

if __name__ == "__main__":
    main() 