#!/usr/bin/env python3
"""Test script to verify health endpoints are properly registered."""

from app.main import app

def test_routes():
    """Test that health routes are properly registered."""
    routes = []
    for route in app.routes:
        if hasattr(route, 'path'):
            routes.append({
                'path': route.path,
                'methods': list(route.methods) if hasattr(route, 'methods') else []
            })
    
    print("🔍 Available routes:")
    health_routes = [r for r in routes if '/health' in r['path'] or r['path'] == '/']
    
    for route in health_routes:
        print(f"  {route['path']} - {route['methods']}")
    
    # Check for required endpoints
    paths = [r['path'] for r in routes]
    required_paths = ['/', '/health', '/api/v1/health']
    
    print("\n✅ Endpoint verification:")
    for path in required_paths:
        if path in paths:
            print(f"  ✅ {path} - Found")
        else:
            print(f"  ❌ {path} - Missing")
    
    # Check for HEAD method support
    print("\n🔍 HEAD method support:")
    for route in routes:
        if route['path'] in ['/', '/health', '/api/v1/health']:
            if 'HEAD' in route['methods']:
                print(f"  ✅ {route['path']} - HEAD supported")
            else:
                print(f"  ⚠️ {route['path']} - HEAD not found in methods: {route['methods']}")

if __name__ == "__main__":
    test_routes() 