"""Core business logic module."""

class CoreModule:
    """Main business logic handler."""
    
    def __init__(self):
        self.state = {}
    
    def process(self, data):
        """Process the provided data."""
        self.state['last_input'] = data
        return {'status': 'success', 'data': data}
    
    def get_state(self):
        """Return current state."""
        return self.state.copy()
