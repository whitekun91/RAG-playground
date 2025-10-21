"""
Image Link Builder - 이미지 링크를 구성하는 클래스
"""
from typing import List


class ImageLinkBuilder:
    """이미지 링크를 구성하는 클래스"""
    
    def build(self, image_refs: List[str]) -> List[str]:
        """이미지 참조를 실제 URL로 변환"""
        if not image_refs:
            return []
        
        # 이미지 참조를 URL로 변환
        image_urls = []
        for ref in image_refs:
            # 간단한 URL 변환 로직
            if ref.startswith('/'):
                # 상대 경로인 경우
                image_urls.append(f"/images{ref}")
            elif ref.startswith('http'):
                # 절대 URL인 경우
                image_urls.append(ref)
            else:
                # 파일명만 있는 경우
                image_urls.append(f"/images/{ref}")
        
        return image_urls
