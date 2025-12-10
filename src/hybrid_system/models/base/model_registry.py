"""
モデルレジストリ

モデルの登録と管理を行う
"""

from typing import Dict, Type, Any, Optional
from .base_model import BaseModel, ModelConfig


class ModelRegistry:
    """
    モデルレジストリ
    
    モデルの登録、検索、インスタンス化を管理
    """
    
    _registry: Dict[str, Type[BaseModel]] = {}
    _configs: Dict[str, ModelConfig] = {}
    
    @classmethod
    def register(cls, name: str, model_class: Type[BaseModel]):
        """モデルを登録
        
        Args:
            name: モデル名
            model_class: モデルクラス
        """
        cls._registry[name] = model_class
        print(f"モデル登録: {name} -> {model_class.__name__}")
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseModel]]:
        """モデルクラスを取得
        
        Args:
            name: モデル名
        
        Returns:
            モデルクラス（見つからない場合はNone）
        """
        return cls._registry.get(name)
    
    @classmethod
    def create(cls, name: str, config: ModelConfig) -> BaseModel:
        """モデルインスタンスを作成
        
        Args:
            name: モデル名
            config: モデル設定
        
        Returns:
            モデルインスタンス
        """
        model_class = cls.get(name)
        if model_class is None:
            raise ValueError(f"モデルが見つかりません: {name}")
        
        return model_class(config)
    
    @classmethod
    def list_models(cls) -> list:
        """登録済みモデルのリストを取得"""
        return list(cls._registry.keys())
    
    @classmethod
    def register_config(cls, name: str, config: ModelConfig):
        """モデル設定を登録
        
        Args:
            name: モデル名
            config: モデル設定
        """
        cls._configs[name] = config
    
    @classmethod
    def get_config(cls, name: str) -> Optional[ModelConfig]:
        """モデル設定を取得
        
        Args:
            name: モデル名
        
        Returns:
            モデル設定（見つからない場合はNone）
        """
        return cls._configs.get(name)
    
    @classmethod
    def clear(cls):
        """レジストリをクリア"""
        cls._registry.clear()
        cls._configs.clear()


def register_model(name: str):
    """モデルを登録するデコレータ
    
    使用例:
        @register_model("my_model")
        class MyModel(BaseModel):
            ...
    """
    def decorator(model_class: Type[BaseModel]):
        ModelRegistry.register(name, model_class)
        return model_class
    return decorator
