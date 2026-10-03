"""WGCNA-style module colour names (labels2colors order) and their R hex values."""

STANDARD = [
    ("turquoise", "#40E0D0"), ("blue", "#0000FF"), ("brown", "#A52A2A"), ("yellow", "#FFFF00"),
    ("green", "#00FF00"), ("red", "#FF0000"), ("black", "#000000"), ("pink", "#FFC0CB"),
    ("magenta", "#FF00FF"), ("purple", "#A020F0"), ("greenyellow", "#ADFF2F"), ("tan", "#D2B48C"),
    ("salmon", "#FA8072"), ("cyan", "#00FFFF"), ("midnightblue", "#191970"), ("lightcyan", "#E0FFFF"),
    ("grey60", "#999999"), ("lightgreen", "#90EE90"), ("lightyellow", "#FFFFE0"), ("royalblue", "#4169E1"),
    ("darkred", "#8B0000"), ("darkgreen", "#006400"), ("darkturquoise", "#00CED1"), ("darkgrey", "#A9A9A9"),
    ("orange", "#FFA500"), ("darkorange", "#FF8C00"), ("white", "#FFFFFF"), ("skyblue", "#87CEEB"),
    ("saddlebrown", "#8B4513"), ("steelblue", "#4682B4"), ("paleturquoise", "#AFEEEE"), ("violet", "#EE82EE"),
    ("darkolivegreen", "#556B2F"), ("darkmagenta", "#8B008B"), ("sienna", "#A0522D"),
    ("yellowgreen", "#9ACD32"), ("skyblue3", "#6CA6CD"), ("plum1", "#FFBBFF"), ("orangered4", "#8B2500"),
    ("mediumpurple3", "#8968CD"), ("lightsteelblue1", "#CAE1FF"), ("lightcyan1", "#E0FFFF"),
    ("ivory", "#FFFFF0"), ("floralwhite", "#FFFAF0"), ("darkorange2", "#EE7600"), ("brown4", "#8B2323"),
    ("bisque4", "#8B7D6B"), ("darkslateblue", "#483D8B"), ("plum2", "#EEAEEE"), ("thistle2", "#EED2EE"),
    ("thistle1", "#FFE1FF"), ("salmon4", "#8B4C39"), ("palevioletred3", "#CD6889"), ("royalblue3", "#3A5FCD"),
]
GREY = ("grey", "#BEBEBE")
HEX = dict(STANDARD)
HEX[GREY[0]] = GREY[1]


def module_color_name(rank: int) -> str:
    """rank 1 = largest module."""
    if rank <= len(STANDARD):
        return STANDARD[rank - 1][0]
    return f"module{rank}"


def hex_for(name: str) -> str:
    if name in HEX:
        return HEX[name]
    # deterministic fallback colour for overflow modules
    import hashlib
    return "#" + hashlib.md5(name.encode()).hexdigest()[:6].upper()
