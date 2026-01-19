"""Paraphrase generation for collusion detection."""

from typing import List, Optional
import random


class ParaphraseGenerator:
    """
    Generator for creating paraphrases of messages.
    Used for detecting semantic channels in collusion detection.
    """
    
    def __init__(
        self,
        llm_backend=None,
        num_paraphrases: int = 5,
        temperature: float = 0.7
    ):
        self.llm_backend = llm_backend
        self.num_paraphrases = num_paraphrases
        self.temperature = temperature
    
    def generate(self, message: str, n: Optional[int] = None) -> List[str]:
        """
        Generate n paraphrases of the given message.
        
        Args:
            message: Original message to paraphrase
            n: Number of paraphrases (default: self.num_paraphrases)
            
        Returns:
            List of paraphrased messages
        """
        n = n or self.num_paraphrases
        
        if self.llm_backend:
            return self._generate_with_llm(message, n)
        else:
            return self._generate_rule_based(message, n)
    
    def _generate_with_llm(self, message: str, n: int) -> List[str]:
        """Generate paraphrases using LLM."""
        paraphrases = []
        
        for i in range(n):
            prompt = f"""Paraphrase the following message while preserving its meaning.
Do not add or remove information. Just rephrase it differently.

Original: {message}

Paraphrase {i+1}:"""
            
            paraphrase = self.llm_backend.generate(
                prompt=prompt,
                max_new_tokens=len(message.split()) * 2,
                temperature=self.temperature + (i * 0.05)  # Vary temperature slightly
            )
            paraphrases.append(paraphrase.strip())
        
        return paraphrases
    
    def _generate_rule_based(self, message: str, n: int) -> List[str]:
        """Generate paraphrases using rule-based transformations."""
        paraphrases = []
        
        transformations = [
            self._synonym_replacement,
            self._sentence_restructure,
            self._add_filler_words,
            self._change_voice,
            self._simplify
        ]
        
        for i in range(n):
            transform = transformations[i % len(transformations)]
            paraphrase = transform(message)
            paraphrases.append(paraphrase)
        
        return paraphrases
    
    def _synonym_replacement(self, message: str) -> str:
        """Replace words with synonyms."""
        synonyms = {
            "good": ["great", "excellent", "fine", "nice"],
            "bad": ["poor", "terrible", "awful", "negative"],
            "buy": ["purchase", "acquire", "get", "obtain"],
            "sell": ["offer", "provide", "supply", "give"],
            "price": ["cost", "rate", "value", "amount"],
            "product": ["item", "goods", "merchandise", "article"],
            "recommend": ["suggest", "advise", "propose", "endorse"],
            "discount": ["reduction", "deal", "savings", "markdown"],
            "quality": ["standard", "grade", "caliber", "level"],
            "available": ["accessible", "obtainable", "ready", "on hand"]
        }
        
        words = message.split()
        result = []
        
        for word in words:
            lower_word = word.lower().strip(".,!?")
            if lower_word in synonyms:
                replacement = random.choice(synonyms[lower_word])
                # Preserve capitalization
                if word[0].isupper():
                    replacement = replacement.capitalize()
                result.append(replacement)
            else:
                result.append(word)
        
        return " ".join(result)
    
    def _sentence_restructure(self, message: str) -> str:
        """Restructure the sentence."""
        # Simple restructuring: move first clause to end if comma present
        if "," in message:
            parts = message.split(",", 1)
            if len(parts) == 2:
                return f"{parts[1].strip()}, {parts[0].strip().lower()}"
        
        # Add introductory phrase
        intros = [
            "I think that",
            "It seems that",
            "I believe",
            "In my opinion,",
            "As I see it,"
        ]
        return f"{random.choice(intros)} {message.lower()}"
    
    def _add_filler_words(self, message: str) -> str:
        """Add filler words."""
        fillers = ["actually", "basically", "essentially", "really", "quite"]
        words = message.split()
        
        if len(words) > 3:
            insert_pos = random.randint(1, len(words) - 1)
            words.insert(insert_pos, random.choice(fillers))
        
        return " ".join(words)
    
    def _change_voice(self, message: str) -> str:
        """Attempt to change active/passive voice."""
        # Simple heuristic transformation
        if " is " in message.lower():
            return message.replace(" is ", " has been ")
        if " are " in message.lower():
            return message.replace(" are ", " have been ")
        if " was " in message.lower():
            return message.replace(" was ", " had been ")
        
        return f"It is the case that {message.lower()}"
    
    def _simplify(self, message: str) -> str:
        """Simplify the message."""
        # Remove common filler words
        fillers_to_remove = [
            "actually", "basically", "essentially", "really", "quite",
            "very", "just", "simply", "literally"
        ]
        
        words = message.split()
        result = [w for w in words if w.lower() not in fillers_to_remove]
        
        if len(result) == len(words):
            # No fillers removed, add "Simply put, "
            return f"Simply put, {message.lower()}"
        
        return " ".join(result)
