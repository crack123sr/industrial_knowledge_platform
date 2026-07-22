import axios from 'axios';

const API_BASE_URL = '/api';

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
      equipment_id: equipmentId || null,
    });
    return response.data;
  },

  uploadDocument: async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await axios.post(`${API_BASE_URL}/documents/upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }
};