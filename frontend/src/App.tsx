import React, { useState } from 'react';
import { copilotAPI } from './apiService';
import { Bot, User, Search, Loader2 } from 'lucide-react';

interface ChatMessage {
  role: 'user' | 'agent';
  content: string;
  sources?: string[];
}

function App() {
  const [query, setQuery] = useState('');
  const [equipmentId, setEquipmentId] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    const userMessage = query;
    setMessages((prev) => [...prev, { role: 'user', content: userMessage }]);
    setQuery('');
    setIsLoading(true);

    try {
      // Call the FastAPI backend
      const result = await copilotAPI.query(userMessage, equipmentId || undefined);
      
      setMessages((prev) => [
        ...prev,
        {
          role: 'agent',
          content: result.answer,
          sources: result.sources,
        },
      ]);
    } catch (error) {
      console.error("API Error:", error);
      setMessages((prev) => [
        ...prev,
        { role: 'agent', content: 'Sorry, I encountered an error connecting to the knowledge base.' },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center p-6 font-sans">
      <header className="w-full max-w-4xl mb-8">
        <h1 className="text-3xl font-bold text-slate-800">Industrial Knowledge Copilot</h1>
        <p className="text-slate-600">Query P&IDs, SOPs, and Maintenance Logs</p>
      </header>

      {/* Main Chat Area */}
      <main className="w-full max-w-4xl bg-white rounded-xl shadow-lg border border-slate-200 overflow-hidden flex flex-col h-[70vh]">
        <div className="flex-1 overflow-y-auto p-6 space-y-6 bg-slate-50">
          {messages.length === 0 ? (
            <div className="flex items-center justify-center h-full text-slate-400">
              Start by asking a question about a procedure or asset...
            </div>
          ) : (
            messages.map((msg, idx) => (
              <div key={idx} className={`flex gap-4 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                {msg.role === 'agent' && <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center text-white shrink-0"><Bot size={18}/></div>}
                
                <div className={`p-4 rounded-xl max-w-[80%] shadow-sm ${msg.role === 'user' ? 'bg-slate-800 text-white rounded-tr-none' : 'bg-white border border-slate-200 rounded-tl-none'}`}>
                  <p className="whitespace-pre-wrap">{msg.content}</p>
                  
                  {/* Render Sources if available */}
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-slate-100">
                      <p className="text-xs font-semibold text-slate-500 mb-1">Sources Cited:</p>
                      <ul className="text-xs text-blue-600 list-disc list-inside">
                        {msg.sources.map((source, i) => (
                          <li key={i}>{source}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
                
                {msg.role === 'user' && <div className="w-8 h-8 rounded-full bg-slate-300 flex items-center justify-center text-slate-700 shrink-0"><User size={18}/></div>}
              </div>
            ))
          )}
          {isLoading && (
            <div className="flex gap-4">
              <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center text-white shrink-0">
                <Loader2 size={18} className="animate-spin" />
              </div>
              <div className="p-4 rounded-xl bg-white border border-slate-200 rounded-tl-none text-slate-500 italic flex items-center gap-2 shadow-sm">
                Searching knowledge graph...
              </div>
            </div>
          )}
        </div>

        {/* Input Form */}
        <div className="p-4 bg-white border-t border-slate-200">
          <form onSubmit={handleSearch} className="flex gap-3">
            <input
              type="text"
              placeholder="Optional: Asset Tag (e.g. P-202A)"
              value={equipmentId}
              onChange={(e) => setEquipmentId(e.target.value)}
              className="w-1/4 px-4 py-3 rounded-lg border border-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
            />
            <div className="relative flex-1">
              <input
                type="text"
                placeholder="Ask about maintenance procedures, safety protocols, or recent failures..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="w-full px-4 py-3 rounded-lg border border-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-500 pl-11 shadow-sm"
                disabled={isLoading}
              />
              <Search className="absolute left-4 top-3.5 text-slate-400" size={18} />
            </div>
            <button
              type="submit"
              disabled={isLoading || !query.trim()}
              className="px-6 py-3 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 disabled:opacity-70 transition-colors shadow-sm"
            >
              Ask Copilot
            </button>
          </form>
        </div>
      </main>
    </div>
  );
}

export default App;