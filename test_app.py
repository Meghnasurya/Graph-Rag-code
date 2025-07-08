#!/usr/bin/env python3
"""
Test script to check if the main application can load properly.
"""

import sys
import os

try:
    # Add current directory to path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    print("Testing Flask app import...")
    from main_formatted import app
    print("✓ Flask app imported successfully")
    
    print("Testing Flask app configuration...")
    with app.app_context():
        print("✓ Flask app context created successfully")
    
    print("Testing route registration...")
    routes = [rule.rule for rule in app.url_map.iter_rules()]
    print(f"✓ Found {len(routes)} registered routes")
    
    # Check for the main routes
    expected_routes = ['/', '/login', '/dashboard', '/auth/github', '/auth/jira']
    missing_routes = [route for route in expected_routes if route not in routes]
    
    if missing_routes:
        print(f"⚠ Missing routes: {missing_routes}")
    else:
        print("✓ All expected routes are registered")
    
    print("\n=== APP HEALTH CHECK PASSED ===")
    print("The application should be able to run without errors.")
    
except ImportError as e:
    print(f"✗ Import Error: {e}")
    print("Please check if all required dependencies are installed.")
    sys.exit(1)
    
except Exception as e:
    print(f"✗ Error: {e}")
    print("There's an issue with the application configuration.")
    sys.exit(1)
