def calculate_distance(loc1, loc2):
    """
    Calcule la "distance" entre deux lieux (0 si identiques, 1 sinon).
    
    Args:
        loc1, loc2: str (quartiers ou lieux)
    
    Returns:
        float: 0.0 si même lieu, 1.0 sinon
    """
    if loc1 is None or loc2 is None:
        return float('inf')
    
    # CORRECTION : Comparaison stricte str uniquement
    if not isinstance(loc1, str) or not isinstance(loc2, str):
        raise TypeError(f"Lieux doivent être des str : loc1={type(loc1)}, loc2={type(loc2)}")
    
    return 0.0 if loc1 == loc2 else 1.0