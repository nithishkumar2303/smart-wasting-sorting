import React, { useState, useEffect } from 'react';
import { LineChart, Line, AreaChart, Area, PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Activity, Trash2, Users, MapPin, TrendingUp, Award, AlertTriangle, CheckCircle } from 'lucide-react';

// Mock API functions (replace with real API calls)
const mockApi = {
  getStats: async () => ({
    total_events: 1247,
    events_by_class: { biodegradable: 456, recyclable: 523, landfill: 268 },
    accuracy_rate: 87.3,
    active_devices: 12,
    total_users: 89,
    contamination_rate: 12.7
  }),
  
  getEvents: async (limit = 10) => Array.from({ length: limit }, (_, i) => ({
    id: i + 1,
    device_id: `device_${(i % 5) + 1}`,
    timestamp: new Date(Date.now() - i * 300000).toISOString(),
    predicted_class: ['biodegradable', 'recyclable', 'landfill'][i % 3],
    confidence: 0.7 + Math.random() * 0.3,
    low_confidence: Math.random() > 0.8
  })),
  
  getTrends: async () => Array.from({ length: 7 }, (_, i) => ({
    date: new Date(Date.now() - (6 - i) * 86400000).toLocaleDateString(),
    biodegradable: Math.floor(Math.random() * 100) + 50,
    recyclable: Math.floor(Math.random() * 120) + 60,
    landfill: Math.floor(Math.random() * 80) + 30
  })),
  
  getLeaderboard: async () => Array.from({ length: 5 }, (_, i) => ({
    user_id: `user_${i + 1}`,
    username: [`EcoWarrior`, `GreenGuru`, `RecycleKing`, `EarthSaver`, `WasteWise`][i],
    total_points: Math.floor(Math.random() * 1000) + 200
  })).sort((a, b) => b.total_points - a.total_points),
  
  getDevices: async () => Array.from({ length: 5 }, (_, i) => ({
    device_id: `device_${i + 1}`,
    location_lat: 13.0827 + (Math.random() - 0.5) * 0.1,
    location_lng: 80.2707 + (Math.random() - 0.5) * 0.1,
    last_seen: new Date(Date.now() - Math.random() * 3600000).toISOString(),
    active: Math.random() > 0.2,
    total_classifications: Math.floor(Math.random() * 200) + 50,
    avg_confidence: 0.7 + Math.random() * 0.25
  }))
};

// Components
const StatCard = ({ title, value, icon: Icon, color = 'blue', change, unit = '' }) => (
  <div className="bg-white rounded-lg shadow-md p-6 border-l-4" style={{ borderLeftColor: color }}>
    <div className="flex items-center justify-between">
      <div>
        <p className="text-sm font-medium text-gray-600">{title}</p>
        <p className="text-2xl font-bold text-gray-900">
          {value}{unit}
        </p>
        {change && (
          <p className={`text-xs ${change > 0 ? 'text-green-600' : 'text-red-600'}`}>
            {change > 0 ? '+' : ''}{change}% from last week
          </p>
        )}
      </div>
      <div className="p-3 rounded-full" style={{ backgroundColor: `${color}20` }}>
        <Icon size={24} style={{ color }} />
      </div>
    </div>
  </div>
);

const EventsTable = ({ events }) => (
  <div className="bg-white rounded-lg shadow-md p-6">
    <h3 className="text-lg font-semibold text-gray-800 mb-4">Recent Classifications</h3>
    <div className="overflow-x-auto">
      <table className="min-w-full table-auto">
        <thead>
          <tr className="border-b border-gray-200">
            <th className="text-left py-3 px-4 font-semibold text-sm text-gray-600">Device</th>
            <th className="text-left py-3 px-4 font-semibold text-sm text-gray-600">Class</th>
            <th className="text-left py-3 px-4 font-semibold text-sm text-gray-600">Confidence</th>
            <th className="text-left py-3 px-4 font-semibold text-sm text-gray-600">Time</th>
            <th className="text-left py-3 px-4 font-semibold text-sm text-gray-600">Status</th>
          </tr>
        </thead>
        <tbody>
          {events.map((event, index) => (
            <tr key={index} className="border-b border-gray-100 hover:bg-gray-50">
              <td className="py-3 px-4 text-sm text-gray-800">{event.device_id}</td>
              <td className="py-3 px-4">
                <span className={`inline-block px-2 py-1 text-xs font-semibold rounded-full ${
                  event.predicted_class === 'biodegradable' ? 'bg-green-100 text-green-800' :
                  event.predicted_class === 'recyclable' ? 'bg-blue-100 text-blue-800' :
                  'bg-gray-100 text-gray-800'
                }`}>
                  {event.predicted_class}
                </span>
              </td>
              <td className="py-3 px-4 text-sm text-gray-800">
                {(event.confidence * 100).toFixed(1)}%
              </td>
              <td className="py-3 px-4 text-sm text-gray-600">
                {new Date(event.timestamp).toLocaleTimeString()}
              </td>
              <td className="py-3 px-4">
                {event.low_confidence ? (
                  <AlertTriangle size={16} className="text-yellow-500" />
                ) : (
                  <CheckCircle size={16} className="text-green-500" />
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  </div>
);

const LeaderboardCard = ({ leaderboard }) => (
  <div className="bg-white rounded-lg shadow-md p-6">
    <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
      <Award className="mr-2 text-yellow-500" size={20} />
      Leaderboard
    </h3>
    <div className="space-y-3">
      {leaderboard.map((user, index) => (
        <div key={user.user_id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
          <div className="flex items-center">
            <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold text-white ${
              index === 0 ? 'bg-yellow-500' :
              index === 1 ? 'bg-gray-400' :
              index === 2 ? 'bg-orange-400' : 'bg-blue-500'
            }`}>
              {index + 1}
            </div>
            <span className="ml-3 font-medium text-gray-800">{user.username}</span>
          </div>
          <span className="font-bold text-blue-600">{user.total_points} pts</span>
        </div>
      ))}
    </div>
  </div>
);

const DeviceStatus = ({ devices }) => (
  <div className="bg-white rounded-lg shadow-md p-6">
    <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
      <MapPin className="mr-2 text-green-500" size={20} />
      Device Status
    </h3>
    <div className="grid grid-cols-1 gap-4">
      {devices.map((device) => (
        <div key={device.device_id} className="border border-gray-200 rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="font-medium text-gray-800">{device.device_id}</span>
            <span className={`px-2 py-1 text-xs font-semibold rounded-full ${
              device.active ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
            }`}>
              {device.active ? 'Active' : 'Offline'}
            </span>
          </div>
          <div className="text-sm text-gray-600 space-y-1">
            <div>Classifications: {device.total_classifications}</div>
            <div>Avg. Confidence: {(device.avg_confidence * 100).toFixed(1)}%</div>
            <div>Last seen: {new Date(device.last_seen).toLocaleString()}</div>
          </div>
        </div>
      ))}
    </div>
  </div>
);

// Main Dashboard Component
export default function SmartWasteDashboard() {
  const [stats, setStats] = useState(null);
  const [events, setEvents] = useState([]);
  const [trends, setTrends] = useState([]);
  const [leaderboard, setLeaderboard] = useState([]);
  const [devices, setDevices] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [statsData, eventsData, trendsData, leaderboardData, devicesData] = await Promise.all([
          mockApi.getStats(),
          mockApi.getEvents(10),
          mockApi.getTrends(),
          mockApi.getLeaderboard(),
          mockApi.getDevices()
        ]);

        setStats(statsData);
        setEvents(eventsData);
        setTrends(trendsData);
        setLeaderboard(leaderboardData);
        setDevices(devicesData);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching data:', error);
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="text-center">
          <Activity className="animate-spin mx-auto mb-4" size={48} />
          <p className="text-gray-600">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  const pieData = stats ? Object.entries(stats.events_by_class).map(([name, value]) => ({
    name: name.charAt(0).toUpperCase() + name.slice(1),
    value,
    color: name === 'biodegradable' ? '#22c55e' : name === 'recyclable' ? '#3b82f6' : '#6b7280'
  })) : [];

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <Trash2 className="text-green-600 mr-3" size={32} />
              <h1 className="text-2xl font-bold text-gray-800">Smart Waste Sorting Dashboard</h1>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center text-sm text-gray-600">
                <div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
                System Online
              </div>
              <div className="text-sm text-gray-600">
                Last updated: {new Date().toLocaleTimeString()}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="bg-white border-b">
        <div className="px-6">
          <nav className="flex space-x-8">
            {[
              { id: 'overview', label: 'Overview' },
              { id: 'analytics', label: 'Analytics' },
              { id: 'devices', label: 'Devices' },
              { id: 'users', label: 'Users' }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`py-4 px-1 border-b-2 font-medium text-sm ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </div>
      </div>

      {/* Main Content */}
      <div className="px-6 py-6">
        {activeTab === 'overview' && (
          <>
            {/* Stats Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
              <StatCard
                title="Total Classifications"
                value={stats?.total_events.toLocaleString() || '0'}
                icon={Activity}
                color="#3b82f6"
                change={8.2}
              />
              <StatCard
                title="Accuracy Rate"
                value={stats?.accuracy_rate || 0}
                icon={CheckCircle}
                color="#22c55e"
                unit="%"
                change={2.1}
              />
              <StatCard
                title="Active Devices"
                value={stats?.active_devices || 0}
                icon={MapPin}
                color="#f59e0b"
                change={-1.2}
              />
              <StatCard
                title="Contamination Rate"
                value={stats?.contamination_rate || 0}
                icon={AlertTriangle}
                color="#ef4444"
                unit="%"
                change={-3.4}
              />
            </div>

            {/* Charts Row */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
              {/* Waste Composition Pie Chart */}
              <div className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-semibold text-gray-800 mb-4">Waste Composition</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={pieData}
                      cx="50%"
                      cy="50%"
                      outerRadius={100}
                      fill="#8884d8"
                      dataKey="value"
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    >
                      {pieData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>

              {/* Weekly Trends */}
              <div className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-semibold text-gray-800 mb-4">Weekly Trends</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={trends}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="biodegradable" stroke="#22c55e" strokeWidth={2} />
                    <Line type="monotone" dataKey="recyclable" stroke="#3b82f6" strokeWidth={2} />
                    <Line type="monotone" dataKey="landfill" stroke="#6b7280" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Recent Events Table */}
            <EventsTable events={events} />
          </>
        )}

        {activeTab === 'analytics' && (
          <div className="space-y-6">
            {/* Detailed Analytics Charts */}
            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">Daily Classification Volume</h3>
              <ResponsiveContainer width="100%" height={400}>
                <AreaChart data={trends}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Area type="monotone" dataKey="biodegradable" stackId="1" stroke="#22c55e" fill="#22c55e" fillOpacity={0.6} />
                  <Area type="monotone" dataKey="recyclable" stackId="1" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.6} />
                  <Area type="monotone" dataKey="landfill" stackId="1" stroke="#6b7280" fill="#6b7280" fillOpacity={0.6} />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-semibold text-gray-800 mb-4">Device Performance</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={devices.map(d => ({ ...d, name: d.device_id }))}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="total_classifications" fill="#3b82f6" />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-semibold text-gray-800 mb-4">Confidence Distribution</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={[
                    { range: '90-100%', count: Math.floor(stats?.total_events * 0.4) },
                    { range: '80-89%', count: Math.floor(stats?.total_events * 0.3) },
                    { range: '70-79%', count: Math.floor(stats?.total_events * 0.2) },
                    { range: '60-69%', count: Math.floor(stats?.total_events * 0.1) }
                  ]}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="range" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="count" fill="#22c55e" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'devices' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <DeviceStatus devices={devices} />
              
              <div className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-semibold text-gray-800 mb-4">Device Locations Map</h3>
                <div className="h-80 bg-gray-100 rounded-lg flex items-center justify-center">
                  <div className="text-center text-gray-500">
                    <MapPin size={48} className="mx-auto mb-2 opacity-50" />
                    <p>Interactive map would be displayed here</p>
                    <p className="text-sm">Showing {devices.filter(d => d.active).length} active devices</p>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">Device Performance Metrics</h3>
              <div className="overflow-x-auto">
                <table className="min-w-full table-auto">
                  <thead>
                    <tr className="border-b border-gray-200">
                      <th className="text-left py-3 px-4 font-semibold text-sm text-gray-600">Device ID</th>
                      <th className="text-left py-3 px-4 font-semibold text-sm text-gray-600">Status</th>
                      <th className="text-left py-3 px-4 font-semibold text-sm text-gray-600">Classifications</th>
                      <th className="text-left py-3 px-4 font-semibold text-sm text-gray-600">Avg Confidence</th>
                      <th className="text-left py-3 px-4 font-semibold text-sm text-gray-600">Last Seen</th>
                      <th className="text-left py-3 px-4 font-semibold text-sm text-gray-600">Uptime</th>
                    </tr>
                  </thead>
                  <tbody>
                    {devices.map((device) => (
                      <tr key={device.device_id} className="border-b border-gray-100 hover:bg-gray-50">
                        <td className="py-3 px-4 text-sm font-medium text-gray-800">{device.device_id}</td>
                        <td className="py-3 px-4">
                          <span className={`inline-block px-2 py-1 text-xs font-semibold rounded-full ${
                            device.active ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                          }`}>
                            {device.active ? 'Online' : 'Offline'}
                          </span>
                        </td>
                        <td className="py-3 px-4 text-sm text-gray-800">{device.total_classifications}</td>
                        <td className="py-3 px-4 text-sm text-gray-800">
                          {(device.avg_confidence * 100).toFixed(1)}%
                        </td>
                        <td className="py-3 px-4 text-sm text-gray-600">
                          {new Date(device.last_seen).toLocaleString()}
                        </td>
                        <td className="py-3 px-4 text-sm text-gray-800">
                          {device.active ? '99.2%' : '0%'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'users' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <LeaderboardCard leaderboard={leaderboard} />
              
              <div className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
                  <Users className="mr-2 text-blue-500" size={20} />
                  User Engagement
                </h3>
                <div className="space-y-4">
                  <div className="flex justify-between items-center p-3 bg-blue-50 rounded-lg">
                    <span className="text-sm font-medium text-gray-700">Total Active Users</span>
                    <span className="text-lg font-bold text-blue-600">{stats?.total_users}</span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-green-50 rounded-lg">
                    <span className="text-sm font-medium text-gray-700">Daily Active Users</span>
                    <span className="text-lg font-bold text-green-600">{Math.floor((stats?.total_users || 0) * 0.3)}</span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-yellow-50 rounded-lg">
                    <span className="text-sm font-medium text-gray-700">Points Distributed</span>
                    <span className="text-lg font-bold text-yellow-600">
                      {leaderboard.reduce((sum, user) => sum + user.total_points, 0).toLocaleString()}
                    </span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-purple-50 rounded-lg">
                    <span className="text-sm font-medium text-gray-700">Avg Points/User</span>
                    <span className="text-lg font-bold text-purple-600">
                      {Math.floor(leaderboard.reduce((sum, user) => sum + user.total_points, 0) / leaderboard.length)}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">User Activity Timeline</h3>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={trends.map((day, index) => ({
                  ...day,
                  active_users: Math.floor(Math.random() * 30) + 10,
                  new_users: Math.floor(Math.random() * 5) + 1
                }))}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Area type="monotone" dataKey="active_users" stackId="1" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.6} />
                  <Area type="monotone" dataKey="new_users" stackId="1" stroke="#22c55e" fill="#22c55e" fillOpacity={0.6} />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">Recent User Activity</h3>
              <div className="space-y-3">
                {Array.from({ length: 10 }, (_, i) => ({
                  user: leaderboard[i % leaderboard.length]?.username || `User${i + 1}`,
                  action: ['Classified recyclable item', 'Earned 10 points', 'Completed daily challenge', 'Reported incorrect classification'][i % 4],
                  time: new Date(Date.now() - i * 600000).toLocaleString(),
                  points: [10, 5, 25, 15][i % 4]
                })).map((activity, index) => (
                  <div key={index} className="flex items-center justify-between p-3 border border-gray-200 rounded-lg">
                    <div className="flex items-center">
                      <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center text-white text-sm font-bold">
                        {activity.user.charAt(0)}
                      </div>
                      <div className="ml-3">
                        <p className="text-sm font-medium text-gray-800">{activity.user}</p>
                        <p className="text-xs text-gray-600">{activity.action}</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="text-sm font-bold text-green-600">+{activity.points} pts</p>
                      <p className="text-xs text-gray-500">{activity.time}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="bg-white border-t mt-8">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between text-sm text-gray-600">
            <div>
              <span className="font-medium">Smart Waste Sorting System</span> - Hackathon 2025
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center">
                <div className="w-2 h-2 bg-green-500 rounded-full mr-2 animate-pulse"></div>
                Live Data
              </div>
              <div>
                {stats && `${stats.total_events} total classifications processed`}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}