"""
Image Link Builder - Convert image references to URLs
"""
from typing import List
import os


class ImageLinkBuilder:
    """Class for converting image references to URLs"""
    
    def __init__(self):
        pass
    
    def build(self, image_refs: List[str]) -> List[str]:
        """
        Convert image references to URLs
        
        Args:
            image_refs: List of image references
            
        Returns:
            List of image URLs
        """
        if not image_refs:
            return []
        
        image_urls = []
        for ref in image_refs:
            if ref:
                # Convert image reference to URL
                # Example: "doc1/page1.png" -> "/images/doc1/page1.png"
                if not ref.startswith('/'):
                    ref = f"/images/{ref}"
                image_urls.append(ref)
        
        return image_urls
