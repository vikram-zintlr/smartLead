from django.http import JsonResponse
from django.core.cache import cache
from datetime import datetime, timedelta
import json
import time
import numpy as np

class APIRequestMiddleware:
    # Define endpoint-specific daily limits
    ENDPOINT_LIMITS = {
        '/api/search/': 1000,  # 1000 requests per day
        '/api/search-by-linkedin/': 500,  # 500 requests per day
        '/api/stats/': 2000,  # 2000 requests per day
        'default': 1000  # Default limit for undefined endpoints
    }

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Only track API endpoints
        if request.path.startswith('/api/'):
            return self.process_api_request(request)
        return self.get_response(request)

    def process_api_request(self, request):
        # Get client IP and start time
        client_ip = self.get_client_ip(request)
        start_time = time.time()
        
        # Get today's date as string for the cache key
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Create cache keys
        daily_key = f"api_requests:{client_ip}:{today}"
        total_key = f"total_api_requests:{today}"
        ip_list_key = f"api_ip_list:{today}"
        endpoint_key = f"api_endpoint:{request.path}:{today}"
        endpoint_list_key = f"api_endpoint_list:{today}"
        endpoint_total_key = f"api_endpoint_total:{request.path}:{today}"
        
        # Keys for additional metrics
        endpoint_times_key = f"api_endpoint_times:{request.path}:{today}"
        endpoint_success_key = f"api_endpoint_success:{request.path}:{today}"
        endpoint_errors_key = f"api_endpoint_errors:{request.path}:{today}"
        endpoint_last_request_key = f"api_endpoint_last_request:{request.path}:{today}"
        endpoint_peak_hour_key = f"api_endpoint_peak_hour:{request.path}:{today}"
        endpoint_min_time_key = f"api_endpoint_min_time:{request.path}:{today}"
        endpoint_max_time_key = f"api_endpoint_max_time:{request.path}:{today}"
        
        # Get current counts from cache
        daily_count = cache.get(daily_key, 0)
        total_count = cache.get(total_key, 0)
        endpoint_count = cache.get(endpoint_key, 0)
        endpoint_total = cache.get(endpoint_total_key, 0)
        
        # Get or initialize additional metrics
        response_times = cache.get(endpoint_times_key, [])
        success_count = cache.get(endpoint_success_key, 0)
        error_count = cache.get(endpoint_errors_key, 0)
        peak_hours = cache.get(endpoint_peak_hour_key, {})
        min_time = cache.get(endpoint_min_time_key, float('inf'))
        max_time = cache.get(endpoint_max_time_key, 0)
        
        # Get or create IP set for today
        ip_list = cache.get(ip_list_key, set())
        ip_list.add(client_ip)
        
        # Get or create endpoint set for today
        endpoint_list = cache.get(endpoint_list_key, set())
        endpoint_list.add(request.path)
        
        # Get endpoint-specific daily limit
        daily_limit = self.ENDPOINT_LIMITS.get(request.path, self.ENDPOINT_LIMITS['default'])
        endpoint_limit = self.ENDPOINT_LIMITS.get(request.path, self.ENDPOINT_LIMITS['default'])
        
        # Check if daily IP limit exceeded
        if daily_count >= daily_limit:
            return JsonResponse({
                'error': 'Daily API request limit exceeded for your IP. Please try again tomorrow.',
                'current_count': daily_count,
                'limit': daily_limit
            }, status=429)
        
        # Check if endpoint total limit exceeded
        if endpoint_total >= endpoint_limit:
            return JsonResponse({
                'error': f'Daily limit for {request.path} endpoint exceeded. Please try again tomorrow.',
                'current_count': endpoint_total,
                'limit': endpoint_limit
            }, status=429)
        
        # Process the request and capture response
        response = self.get_response(request)
        end_time = time.time()
        response_time = end_time - start_time
        
        # Update metrics based on response
        is_success = 200 <= response.status_code < 400
        if is_success:
            success_count += 1
        else:
            error_count += 1
        
        # Update response times (keep last 1000 times for average calculation)
        response_times.append(response_time)
        if len(response_times) > 1000:
            response_times = response_times[-1000:]
        
        # Update min/max times
        min_time = min(min_time, response_time)
        max_time = max(max_time, response_time)
        
        # Update peak hour tracking
        current_hour = datetime.now().hour
        peak_hours[current_hour] = peak_hours.get(current_hour, 0) + 1
        
        # Increment counters and update metrics
        cache.set(daily_key, daily_count + 1, timeout=86400)  # 24 hours timeout
        cache.set(total_key, total_count + 1, timeout=86400)
        cache.set(ip_list_key, ip_list, timeout=86400)
        cache.set(endpoint_key, endpoint_count + 1, timeout=86400)
        cache.set(endpoint_list_key, endpoint_list, timeout=86400)
        cache.set(endpoint_total_key, endpoint_total + 1, timeout=86400)
        
        # Update additional metrics
        cache.set(endpoint_times_key, response_times, timeout=86400)
        cache.set(endpoint_success_key, success_count, timeout=86400)
        cache.set(endpoint_errors_key, error_count, timeout=86400)
        cache.set(endpoint_last_request_key, datetime.now().isoformat(), timeout=86400)
        cache.set(endpoint_peak_hour_key, peak_hours, timeout=86400)
        cache.set(endpoint_min_time_key, min_time, timeout=86400)
        cache.set(endpoint_max_time_key, max_time, timeout=86400)
        
        # Add request count headers
        response['X-Daily-Request-Count'] = daily_count + 1
        response['X-Daily-Request-Limit'] = daily_limit
        response['X-Endpoint-Request-Count'] = endpoint_total + 1
        response['X-Endpoint-Request-Limit'] = endpoint_limit
        
        # Log the request with enhanced metrics
        self.log_request(request, client_ip, daily_count + 1, endpoint_total + 1, response_time, is_success)
        
        return response

    def get_client_ip(self, request):
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip

    def log_request(self, request, client_ip, count, endpoint_count, response_time, is_success):
        # Get request body for POST requests
        body = None
        if request.method == 'POST':
            try:
                body = json.loads(request.body)
            except:
                body = None

        log_data = {
            'timestamp': datetime.now().isoformat(),
            'ip': client_ip,
            'path': request.path,
            'method': request.method,
            'ip_count': count,
            'endpoint_count': endpoint_count,
            'endpoint_limit': self.ENDPOINT_LIMITS.get(request.path, self.ENDPOINT_LIMITS['default']),
            'response_time': response_time,
            'success': is_success,
            'body': body
        }
        
        # Log to console for now
        print(f"API Request Log: {log_data}")

    def get_endpoint_stats(self, endpoint_path):
        """Get comprehensive stats for a specific endpoint"""
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Get all relevant metrics from cache
        endpoint_total = cache.get(f"api_endpoint_total:{endpoint_path}:{today}", 0)
        response_times = cache.get(f"api_endpoint_times:{endpoint_path}:{today}", [])
        success_count = cache.get(f"api_endpoint_success:{endpoint_path}:{today}", 0)
        error_count = cache.get(f"api_endpoint_errors:{endpoint_path}:{today}", 0)
        last_request = cache.get(f"api_endpoint_last_request:{endpoint_path}:{today}", None)
        peak_hours = cache.get(f"api_endpoint_peak_hour:{endpoint_path}:{today}", {})
        min_time = cache.get(f"api_endpoint_min_time:{endpoint_path}:{today}", None)
        max_time = cache.get(f"api_endpoint_max_time:{endpoint_path}:{today}", None)
        
        # Calculate metrics
        total_requests = success_count + error_count
        success_rate = (success_count / total_requests) if total_requests > 0 else 1.0
        
        # Calculate response time metrics
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            p95_response_time = np.percentile(response_times, 95) if len(response_times) >= 20 else None
        else:
            avg_response_time = None
            p95_response_time = None
            
        peak_hour = max(peak_hours.items(), key=lambda x: x[1])[0] if peak_hours else None
        
        return {
            'endpoint': endpoint_path,
            'request_count': endpoint_total,
            'metrics': {
                'response_time': {
                    'average': avg_response_time,
                    'min': min_time if min_time is not None and min_time != float('inf') else None,
                    'max': max_time if max_time is not None else None,
                    'p95': p95_response_time
                },
                'success_rate': success_rate,
                'last_request_timestamp': last_request,
                'peak_hour': peak_hour,
                'total_success': success_count,
                'total_errors': error_count,
                'error_rate': 1 - success_rate
            }
        } 