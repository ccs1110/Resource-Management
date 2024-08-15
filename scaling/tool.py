
def convert_resource_value(value):
    """Convert resource value from Kubernetes format to integer."""
    if value is None:
        return 0
    if value.endswith('m'):
        return int(value[:-1])  # Convert millicores to integer
    if value.endswith('Mi'):
        return int(value[:-2])  # Convert Mebibytes to integer
    if value.endswith('Gi'):
        return int(value[:-2]) * 1024  # Convert Gibibytes to Mebibytes
    return int(value)