import secrets

# Generate a random secret key
secret_key = secrets.token_hex(16)  # Generates a 32-character hexadecimal string
print(secret_key)