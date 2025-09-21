// dashboard/src/api.js
import axios from "axios";

const baseURL = process.env.REACT_APP_API_URL || "/api";
export const api = axios.create({ baseURL });

api.interceptors.request.use((config) => {
  config.headers.Authorization = `Bearer ${process.env.REACT_APP_API_TOKEN || "admin_token"}`;
  return config;
});