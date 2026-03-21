"""The simplest possible Python script to run in Docker."""
import sys
import platform

print("=" * 40)
print("Hello from inside a Docker container!")
print("=" * 40)
print(f"Python version: {sys.version}")
print(f"OS:             {platform.system()} {platform.release()}")
print(f"Architecture:   {platform.machine()}")
print()
print("This is NOT your laptop's Python.")
print("This is a completely isolated Linux environment.")
