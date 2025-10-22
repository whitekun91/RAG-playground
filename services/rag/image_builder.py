"""
Image Link Builder - 이미지 참조를 URL로 변환하는 모듈
"""
from typing import List
import os


class ImageLinkBuilder:
    """이미지 참조를 URL로 변환하는 클래스"""
    
    def __init__(self):
        pass
    
    def build(self, image_refs: List[str]) -> List[str]:
        """
        이미지 참조를 URL로 변환
        
        Args:
            image_refs: 이미지 참조 리스트
            
        Returns:
            이미지 URL 리스트
        """
        if not image_refs:
            return []
        
        image_urls = []
        for ref in image_refs:
            if ref:
                # 이미지 참조를 URL로 변환
                # 예: "doc1/page1.png" -> "/images/doc1/page1.png"
                if not ref.startswith('/'):
                    ref = f"/images/{ref}"
                image_urls.append(ref)
        
        return image_urls
