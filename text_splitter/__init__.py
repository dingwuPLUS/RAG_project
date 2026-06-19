"""
分块器工厂接口，根据配置返回对应分块函数。
"""

from .splitters import fixed_size_split, recursive_split, SemanticSplitter


def get_splitter(strategy: str, **kwargs):
    """
    返回一个分块函数，签名：List[Document] -> List[Document]

    strategy: 'fixed', 'recursive', 'semantic'
    kwargs: 相关参数，如 chunk_size, chunk_overlap, embed_fn 等
    """
    if strategy == "fixed":
        return lambda docs: fixed_size_split(
            docs,
            chunk_size=kwargs.get("chunk_size", 500),
            chunk_overlap=kwargs.get("chunk_overlap", 50)
        )
    elif strategy == "recursive":
        return lambda docs: recursive_split(
            docs,
            chunk_size=kwargs.get("chunk_size", 500),
            chunk_overlap=kwargs.get("chunk_overlap", 50),
            separators=kwargs.get("separators", None)
        )
    elif strategy == "semantic":
        embed_fn = kwargs.get("embed_fn")
        if not embed_fn:
            raise ValueError("语义分块必须提供 'embed_fn' 参数（嵌入函数）")
        splitter = SemanticSplitter(
            embed_fn=embed_fn,
            similarity_threshold=kwargs.get("similarity_threshold", 0.8),
            min_sentences=kwargs.get("min_sentences", 1)
        )
        return splitter.split
    else:
        raise ValueError(f"未知分块策略: {strategy}，可选 'fixed', 'recursive', 'semantic'")