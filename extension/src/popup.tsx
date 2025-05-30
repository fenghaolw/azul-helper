import { render } from 'preact';
import App from './components/App';
import './styles.css';

const rootElement = document.getElementById('root');
if (rootElement) {
  render(<App />, rootElement);
}
