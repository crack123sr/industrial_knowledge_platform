import axios from 'axios';

// The base URL of your running FastAPI server
const API_BASE_URL = 'http://localhost:8000/api';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const copilotAPI = {
  query: async (question: string, equipmentId?: string) => {
    const response = await apiClient.post('/copilot/query', {
      query: question,
      equipment_id: equipmentId,
    });
    return response.data;
  },
};