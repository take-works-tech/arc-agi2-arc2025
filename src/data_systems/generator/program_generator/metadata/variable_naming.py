"""
変数命名システム

学習精度向上のため、型ごとに最小限の変数名 + ナンバリングを使用
"""
import logging
from typing import Dict, List, Set, Optional, Union, Any
from .types import SemanticType, ReturnType, TypeInfo

# ログ設定
logger = logging.getLogger(__name__)


class VariableNamingSystem:
    """動的最小限変数命名システム

    学習精度向上のため、型ごとに最小限の変数名パターンを動的に生成
    """

    def __init__(self, max_variables_per_type: int = 10):
        """初期化

        Args:
            max_variables_per_type: 型あたりの最大変数数
        """
        # 型ごとのベース変数名パターン（SemanticTypeに統一）
        self.base_patterns: Dict[SemanticType, List[str]] = {
            # 座標系 - 最も基本的なパターン
            SemanticType.COORDINATE: ['coordinate'],

            # サイズ系 - 汎用的なサイズ変数
            SemanticType.SIZE: ['size'],

            # 色系 - 基本的な色変数
            SemanticType.COLOR: ['color'],

            # 距離系 - 距離変数
            SemanticType.DISTANCE: ['dist'],
            SemanticType.DISTANCE_AXIS: ['x_dist', 'y_dist'],

            # カウント系 - カウント変数（細分化）
            SemanticType.COUNT: ['count'],
            SemanticType.COUNT_HOLES: ['hole_count'],
            SemanticType.COUNT_ADJACENT: ['adj_count'],
            SemanticType.COUNT_OVERLAP: ['overlap_count'],

            # オフセット系 - オフセット変数
            SemanticType.OFFSET: ['dx', 'dy'],

            # パーセンテージ系 - スコア変数
            SemanticType.PERCENTAGE: ['score'],

            # 比率型 - 比率変数（基本値との比較で使用）
            SemanticType.ASPECT_RATIO: ['aspect_ratio'],
            SemanticType.DENSITY: ['density'],

            # 角度系 - 角度変数
            SemanticType.ANGLE: ['angle'],

            # スケール系 - スケール変数
            SemanticType.SCALE: ['scale'],

            # オブジェクト系 - オブジェクト関連変数
            SemanticType.OBJECT: ['object'],         # 単一オブジェクト

            # 文字列型 - 文字列変数
            SemanticType.STRING: ['string'],
            SemanticType.DIRECTION: ['direction'],
            SemanticType.AXIS: ['axis'],
            SemanticType.LINE_TYPE: ['line_type'],
            SemanticType.RECT_TYPE: ['rect_type'],
            SemanticType.SIDE: ['side'],
            SemanticType.ORDER: ['order'],
            SemanticType.ALIGN_MODE: ['align_mode'],

            # 真偽値型 - 真偽値変数
            SemanticType.BOOL: ['flag'],

            # 特殊型 - 戻り値なし
            SemanticType.VOID: ['result'],
        }

        # デフォルトの変数名パターン
        self.default_base = 'var'

        # 最大変数数の制限（設定可能）
        self.max_variables_per_type = max_variables_per_type

        # パフォーマンス最適化: 型推定用の逆引き辞書
        self._type_lookup_cache: Dict[str, SemanticType] = {}
        self._build_type_lookup_cache()

        # 変数追跡システム
        self._variable_tracker: Dict[str, Dict] = {}  # 変数名 -> {type_info, position, usage_count, etc.}
        self._position_counter: Dict[SemanticType, int] = {}  # 型ごとの位置カウンター

        logger.info(f"VariableNamingSystem initialized with max_variables_per_type={max_variables_per_type}")

    def _build_type_lookup_cache(self):
        """型推定用のキャッシュを構築"""
        for semantic_type, base_names in self.base_patterns.items():
            for base_name in base_names:
                self._type_lookup_cache[base_name.lower()] = semantic_type

    def get_variable_candidates(self, type_info: TypeInfo, max_candidates: int = 4) -> List[str]:
        """型情報に応じた動的変数名候補を取得

        Args:
            type_info: 型情報（配列かどうかの情報を含む）
            max_candidates: 最大候補数

        Returns:
            変数名候補のリスト
        """
        semantic_type = type_info.semantic_type
        base_names = self.base_patterns.get(semantic_type, [self.default_base])

        candidates = []
        for base_name in base_names:
            # 配列の場合は複数形を考慮
            if type_info.is_array:
                base_name = f"{base_name}s"

            # 基本パターン
            candidates.append(base_name)

            # ナンバリングパターン
            for i in range(1, min(max_candidates, self.max_variables_per_type)):
                candidates.append(f"{base_name}{i}")

        return candidates[:max_candidates]

    def get_next_variable_name(
        self,
        type_info: TypeInfo,
        used_vars: Set[str],
        preferred_name: Optional[str] = None
    ) -> str:
        """次の変数名を取得

        Args:
            type_info: 型情報
            used_vars: 既に使用されている変数名のセット
            preferred_name: 優先する変数名（オプション）

        Returns:
            生成された変数名
        """
        # 優先名が指定されている場合、それを最優先で使用
        if preferred_name:
            # 優先名が使用されていない場合、そのまま返す
            if preferred_name not in used_vars:
                return preferred_name
            # 優先名が既に使用されている場合でも、優先名をベースに番号を付けて使用
            # 優先名の形式: objects1_24 -> objects1, 24
            import re
            match = re.match(r'^(.+?)(\d+)$', preferred_name)
            if match:
                base_name = match.group(1)  # objects1_
                suffix = match.group(2)  # 24
                # 優先名のベース + サフィックス + 番号で試行
                counter = 1
                while counter < 100:  # 最大100回試行
                    candidate = f"{base_name}{int(suffix) + counter}"
                    if candidate not in used_vars:
                        return candidate
                    counter += 1
            # フォールバック処理に進む

        # 型に応じた候補を生成
        candidates = self.get_variable_candidates(type_info)

        # 使用されていない候補を探す
        for candidate in candidates:
            if candidate not in used_vars:
                return candidate

        # フォールバック: 型名 + 番号
        semantic_type_name = type_info.semantic_type.value.lower()
        # 配列型の場合は複数形を考慮
        if type_info.is_array:
            # object -> objects のように複数形に変換
            if semantic_type_name == "object":
                semantic_type_name = "objects"
            elif not semantic_type_name.endswith('s'):
                semantic_type_name = f"{semantic_type_name}s"
        counter = 1
        while True:
            candidate = f"{semantic_type_name}{counter}"
            if candidate not in used_vars:
                return candidate
            counter += 1

    def get_argument_name(self, type_info: TypeInfo) -> str:
        """引数名を取得

        Args:
            type_info: 型情報

        Returns:
            引数名
        """
        semantic_type = type_info.semantic_type
        base_names = self.base_patterns.get(semantic_type, [self.default_base])

        # 最初のベース名を使用
        base_name = base_names[0]

        # 配列の場合は複数形
        if type_info.is_array:
            base_name = f"{base_name}s"

        return base_name

    def get_semantic_type_from_variable_name(self, var_name: str) -> Optional[SemanticType]:
        """変数名から意味的型を推定（最適化版）

        Args:
            var_name: 変数名

        Returns:
            推定された意味的型（該当なしの場合はNone）
        """
        if not var_name or not isinstance(var_name, str):
            logger.warning(f"Invalid variable name: {var_name}")
            return None

        var_name_lower = var_name.lower()

        try:
            # 配列要素参照の特別処理（最優先）
            if '[' in var_name_lower and ']' in var_name_lower:
                base_name = var_name_lower.split('[')[0]
                if base_name == 'grid_size':
                    logger.debug(f"Detected array reference: {var_name} -> {SemanticType.SIZE}")
                    return SemanticType.SIZE
                elif base_name == 'colors':
                    logger.debug(f"Detected array reference: {var_name} -> {SemanticType.COLOR}")
                    return SemanticType.COLOR

            # キャッシュを使用した高速型推定
            if var_name_lower.startswith('colors'):
                logger.debug(f"Type lookup cache hit: {var_name} -> {SemanticType.COLOR}")
                return SemanticType.COLOR
            elif var_name_lower.startswith('coordinates'):
                logger.debug(f"Type lookup cache hit: {var_name} -> {SemanticType.COORDINATE}")
                return SemanticType.COORDINATE
            elif var_name_lower.startswith('sizes'):
                logger.debug(f"Type lookup cache hit: {var_name} -> {SemanticType.SIZE}")
                return SemanticType.SIZE
            elif var_name_lower.startswith('objects'):
                logger.debug(f"Type lookup cache hit: {var_name} -> {SemanticType.OBJECT}")
                return SemanticType.OBJECT

            # ベース名から型を推定
            for base_name, semantic_type in self._type_lookup_cache.items():
                if var_name_lower.startswith(base_name):
                    logger.debug(f"Type lookup cache hit: {var_name} -> {semantic_type}")
                    return semantic_type

            # フォールバック: デフォルト型
            logger.debug(f"No type match found for: {var_name}, using default")
            return SemanticType.OBJECT

        except Exception as e:
            logger.error(f"Error in type inference for {var_name}: {e}")
            return None


# グローバルインスタンス
variable_naming_system = VariableNamingSystem()
