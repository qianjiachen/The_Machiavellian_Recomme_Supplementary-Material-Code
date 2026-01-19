"""Amazon Reviews dataset loader for semi-real validation."""

import json
import gzip
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator
from dataclasses import dataclass
import numpy as np

from src.models.data_models import UserPersona, Item


@dataclass
class AmazonReview:
    """Amazon review data."""
    review_id: str
    user_id: str
    product_id: str
    rating: float
    review_text: str
    summary: str
    timestamp: int
    helpful_votes: int
    verified_purchase: bool


@dataclass
class AmazonProduct:
    """Amazon product data."""
    product_id: str
    title: str
    description: str
    brand: str
    category: List[str]
    price: Optional[float]
    features: List[str]


class AmazonReviewsLoader:
    """
    Loader for Amazon Reviews dataset.
    
    Supports the Amazon Product Reviews dataset format.
    Download from: https://nijianmo.github.io/amazon/index.html
    """
    
    SUPPORTED_CATEGORIES = [
        "Electronics",
        "Books",
        "Clothing_Shoes_and_Jewelry",
        "Home_and_Kitchen",
        "Sports_and_Outdoors"
    ]
    
    def __init__(
        self,
        data_dir: str = "data/amazon",
        category: str = "Electronics",
        embedding_dim: int = 128
    ):
        self.data_dir = Path(data_dir)
        self.category = category
        self.embedding_dim = embedding_dim
        
        self.reviews: List[AmazonReview] = []
        self.products: Dict[str, AmazonProduct] = {}
        self.user_reviews: Dict[str, List[AmazonReview]] = {}
        self.product_reviews: Dict[str, List[AmazonReview]] = {}

    def load_reviews(self, file_path: Optional[str] = None, max_reviews: int = 100000) -> int:
        """
        Load reviews from file.
        
        Args:
            file_path: Path to reviews file (gzipped JSON lines)
            max_reviews: Maximum number of reviews to load
            
        Returns:
            Number of reviews loaded
        """
        if file_path is None:
            file_path = self.data_dir / f"{self.category}.json.gz"
        
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"Reviews file not found: {file_path}")
            print("Please download from: https://nijianmo.github.io/amazon/index.html")
            return 0
        
        count = 0
        opener = gzip.open if str(file_path).endswith('.gz') else open
        
        with opener(file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                if count >= max_reviews:
                    break
                
                try:
                    data = json.loads(line)
                    review = self._parse_review(data)
                    if review:
                        self.reviews.append(review)
                        
                        # Index by user
                        if review.user_id not in self.user_reviews:
                            self.user_reviews[review.user_id] = []
                        self.user_reviews[review.user_id].append(review)
                        
                        # Index by product
                        if review.product_id not in self.product_reviews:
                            self.product_reviews[review.product_id] = []
                        self.product_reviews[review.product_id].append(review)
                        
                        count += 1
                except (json.JSONDecodeError, KeyError):
                    continue
        
        return count
    
    def _parse_review(self, data: Dict) -> Optional[AmazonReview]:
        """Parse a review from JSON data."""
        try:
            return AmazonReview(
                review_id=data.get('reviewerID', '') + '_' + data.get('asin', ''),
                user_id=data.get('reviewerID', ''),
                product_id=data.get('asin', ''),
                rating=float(data.get('overall', 0)),
                review_text=data.get('reviewText', ''),
                summary=data.get('summary', ''),
                timestamp=int(data.get('unixReviewTime', 0)),
                helpful_votes=data.get('helpful', [0, 0])[0] if isinstance(data.get('helpful'), list) else 0,
                verified_purchase=data.get('verified', False)
            )
        except (ValueError, TypeError):
            return None

    def load_products(self, file_path: Optional[str] = None) -> int:
        """
        Load product metadata.
        
        Args:
            file_path: Path to metadata file
            
        Returns:
            Number of products loaded
        """
        if file_path is None:
            file_path = self.data_dir / f"meta_{self.category}.json.gz"
        
        file_path = Path(file_path)
        if not file_path.exists():
            return 0
        
        count = 0
        opener = gzip.open if str(file_path).endswith('.gz') else open
        
        with opener(file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    product = self._parse_product(data)
                    if product:
                        self.products[product.product_id] = product
                        count += 1
                except (json.JSONDecodeError, KeyError):
                    continue
        
        return count
    
    def _parse_product(self, data: Dict) -> Optional[AmazonProduct]:
        """Parse a product from JSON data."""
        try:
            price = data.get('price')
            if isinstance(price, str):
                # Remove $ and commas from price string
                price_str = price.replace('$', '').replace(',', '')
                price = float(price_str) if price_str else None
            
            return AmazonProduct(
                product_id=data.get('asin', ''),
                title=data.get('title', ''),
                description=data.get('description', [''])[0] if isinstance(data.get('description'), list) else str(data.get('description', '')),
                brand=data.get('brand', ''),
                category=data.get('category', []),
                price=price,
                features=data.get('feature', [])
            )
        except (ValueError, TypeError):
            return None
    
    def get_active_users(self, min_reviews: int = 5) -> List[str]:
        """Get users with at least min_reviews reviews."""
        return [
            user_id for user_id, reviews in self.user_reviews.items()
            if len(reviews) >= min_reviews
        ]
    
    def get_popular_products(self, min_reviews: int = 10) -> List[str]:
        """Get products with at least min_reviews reviews."""
        return [
            product_id for product_id, reviews in self.product_reviews.items()
            if len(reviews) >= min_reviews
        ]

    def create_user_persona(self, user_id: str) -> Optional[UserPersona]:
        """
        Create a UserPersona from Amazon review history.
        
        Args:
            user_id: Amazon user ID
            
        Returns:
            UserPersona object or None if user not found
        """
        if user_id not in self.user_reviews:
            return None
        
        reviews = self.user_reviews[user_id]
        
        # Compute preference embedding from review history
        preference_embedding = self._compute_user_embedding(reviews)
        
        # Compute average rating as quality sensitivity
        avg_rating = np.mean([r.rating for r in reviews])
        quality_sensitivity = avg_rating / 5.0
        
        # Estimate price sensitivity from review patterns
        price_sensitivity = self._estimate_price_sensitivity(reviews)
        
        # Compute budget from product prices
        budget = self._estimate_budget(reviews)
        
        return UserPersona(
            user_id=user_id,
            preference_embedding=preference_embedding,
            quality_sensitivity=quality_sensitivity,
            price_sensitivity=price_sensitivity,
            budget=budget
        )
    
    def _compute_user_embedding(self, reviews: List[AmazonReview]) -> np.ndarray:
        """Compute user preference embedding from reviews."""
        # Simple embedding based on rating distribution and review patterns
        embedding = np.zeros(self.embedding_dim)
        
        if not reviews:
            return embedding
        
        # Use rating statistics
        ratings = [r.rating for r in reviews]
        embedding[0] = np.mean(ratings)
        embedding[1] = np.std(ratings) if len(ratings) > 1 else 0
        embedding[2] = len(reviews) / 100.0  # Normalized review count
        
        # Use helpful votes
        helpful = [r.helpful_votes for r in reviews]
        embedding[3] = np.mean(helpful) if helpful else 0
        
        # Use verified purchase ratio
        verified = [1 if r.verified_purchase else 0 for r in reviews]
        embedding[4] = np.mean(verified)
        
        # Fill rest with random but deterministic values based on user
        rng = np.random.RandomState(hash(reviews[0].user_id) % (2**32))
        embedding[5:] = rng.randn(self.embedding_dim - 5) * 0.1
        
        return embedding
    
    def _estimate_price_sensitivity(self, reviews: List[AmazonReview]) -> float:
        """Estimate price sensitivity from review patterns."""
        # Higher ratings for cheaper products = higher price sensitivity
        # This is a simplified heuristic
        if not reviews:
            return 0.5
        
        # Check if user tends to give higher ratings
        avg_rating = np.mean([r.rating for r in reviews])
        
        # Users who rate generously are assumed less price sensitive
        return 1.0 - (avg_rating / 5.0) * 0.5
    
    def _estimate_budget(self, reviews: List[AmazonReview]) -> float:
        """Estimate user budget from reviewed products."""
        prices = []
        for review in reviews:
            if review.product_id in self.products:
                product = self.products[review.product_id]
                if product.price is not None:
                    prices.append(product.price)
        
        if not prices:
            return 100.0  # Default budget
        
        # Use 90th percentile of purchased prices as budget estimate
        return float(np.percentile(prices, 90))

    def create_item(self, product_id: str) -> Optional[Item]:
        """
        Create an Item from Amazon product data.
        
        Args:
            product_id: Amazon product ID (ASIN)
            
        Returns:
            Item object or None if product not found
        """
        product = self.products.get(product_id)
        reviews = self.product_reviews.get(product_id, [])
        
        if product is None and not reviews:
            return None
        
        # Compute feature embedding
        feature_embedding = self._compute_item_embedding(product, reviews)
        
        # Compute quality from reviews
        if reviews:
            quality = np.mean([r.rating for r in reviews]) / 5.0
        else:
            quality = 0.5
        
        # Get price
        price = product.price if product and product.price else 50.0
        
        # Compute popularity
        popularity = min(len(reviews) / 100.0, 1.0)
        
        return Item(
            item_id=product_id,
            feature_embedding=feature_embedding,
            quality=quality,
            price=price,
            popularity=popularity
        )
    
    def _compute_item_embedding(
        self, 
        product: Optional[AmazonProduct], 
        reviews: List[AmazonReview]
    ) -> np.ndarray:
        """Compute item feature embedding."""
        embedding = np.zeros(self.embedding_dim)
        
        # Use product metadata if available
        if product:
            # Category encoding (simple hash-based)
            if product.category:
                cat_hash = hash(tuple(product.category)) % (2**32)
                rng = np.random.RandomState(cat_hash)
                embedding[:10] = rng.randn(10) * 0.5
            
            # Price feature
            if product.price:
                embedding[10] = np.log1p(product.price) / 10.0
            
            # Feature count
            embedding[11] = len(product.features) / 10.0
        
        # Use review statistics
        if reviews:
            ratings = [r.rating for r in reviews]
            embedding[12] = np.mean(ratings) / 5.0
            embedding[13] = np.std(ratings) if len(ratings) > 1 else 0
            embedding[14] = len(reviews) / 100.0
            
            # Helpful votes
            helpful = [r.helpful_votes for r in reviews]
            embedding[15] = np.mean(helpful) / 10.0 if helpful else 0
        
        # Fill rest with deterministic random values
        seed = hash(product.product_id if product else str(reviews[0].product_id if reviews else 'default'))
        rng = np.random.RandomState(seed % (2**32))
        embedding[16:] = rng.randn(self.embedding_dim - 16) * 0.1
        
        return embedding

    def extract_communication_templates(self, n_templates: int = 100) -> Dict[str, List[str]]:
        """
        Extract communication templates from reviews.
        
        These can be used to make LLM-generated messages more realistic.
        
        Args:
            n_templates: Number of templates to extract per category
            
        Returns:
            Dictionary mapping template types to lists of templates
        """
        templates = {
            "positive_review": [],
            "negative_review": [],
            "neutral_review": [],
            "product_inquiry": [],
            "recommendation": []
        }
        
        for review in self.reviews[:n_templates * 5]:
            text = review.review_text.strip()
            if not text or len(text) < 20:
                continue
            
            # Categorize by rating
            if review.rating >= 4.0:
                if len(templates["positive_review"]) < n_templates:
                    templates["positive_review"].append(text)
            elif review.rating <= 2.0:
                if len(templates["negative_review"]) < n_templates:
                    templates["negative_review"].append(text)
            else:
                if len(templates["neutral_review"]) < n_templates:
                    templates["neutral_review"].append(text)
            
            # Extract recommendation-like sentences
            if "recommend" in text.lower() and len(templates["recommendation"]) < n_templates:
                templates["recommendation"].append(text)
            
            # Extract inquiry-like sentences (questions)
            if "?" in text and len(templates["product_inquiry"]) < n_templates:
                templates["product_inquiry"].append(text)
        
        return templates

    def compute_ground_truth_utility(
        self, 
        user_id: str, 
        product_id: str
    ) -> Optional[float]:
        """
        Compute ground truth utility from actual review.
        
        Args:
            user_id: User ID
            product_id: Product ID
            
        Returns:
            Utility value (normalized rating) or None if no review exists
        """
        if user_id not in self.user_reviews:
            return None
        
        for review in self.user_reviews[user_id]:
            if review.product_id == product_id:
                # Normalize rating to [0, 1]
                return review.rating / 5.0
        
        return None
    
    def get_user_product_pairs(
        self, 
        min_user_reviews: int = 5,
        min_product_reviews: int = 10
    ) -> List[Tuple[str, str, float]]:
        """
        Get user-product pairs with ground truth ratings.
        
        Args:
            min_user_reviews: Minimum reviews per user
            min_product_reviews: Minimum reviews per product
            
        Returns:
            List of (user_id, product_id, rating) tuples
        """
        active_users = set(self.get_active_users(min_user_reviews))
        popular_products = set(self.get_popular_products(min_product_reviews))
        
        pairs = []
        for review in self.reviews:
            if review.user_id in active_users and review.product_id in popular_products:
                pairs.append((review.user_id, review.product_id, review.rating))
        
        return pairs
    
    def create_mock_data(self, n_users: int = 100, n_products: int = 50) -> None:
        """
        Create mock data for testing without real dataset.
        
        Args:
            n_users: Number of mock users
            n_products: Number of mock products
        """
        rng = np.random.RandomState(42)
        
        # Create mock products
        for i in range(n_products):
            product_id = f"MOCK_{i:04d}"
            self.products[product_id] = AmazonProduct(
                product_id=product_id,
                title=f"Mock Product {i}",
                description=f"Description for mock product {i}",
                brand=f"Brand_{i % 10}",
                category=[f"Category_{i % 5}"],
                price=rng.uniform(10, 500),
                features=[f"Feature {j}" for j in range(rng.randint(1, 5))]
            )
            self.product_reviews[product_id] = []
        
        # Create mock users and reviews
        product_ids = list(self.products.keys())
        for i in range(n_users):
            user_id = f"USER_{i:04d}"
            self.user_reviews[user_id] = []
            
            # Each user reviews some products
            n_reviews = rng.randint(3, 15)
            reviewed_products = rng.choice(product_ids, size=min(n_reviews, len(product_ids)), replace=False)
            
            for product_id in reviewed_products:
                rating = rng.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.15, 0.3, 0.4])
                review = AmazonReview(
                    review_id=f"{user_id}_{product_id}",
                    user_id=user_id,
                    product_id=product_id,
                    rating=float(rating),
                    review_text=f"Mock review text for product {product_id}",
                    summary=f"Mock summary",
                    timestamp=1600000000 + rng.randint(0, 10000000),
                    helpful_votes=rng.randint(0, 20),
                    verified_purchase=rng.random() > 0.3
                )
                self.reviews.append(review)
                self.user_reviews[user_id].append(review)
                self.product_reviews[product_id].append(review)
